import re
# import os
# import json
from graphviz import Digraph
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
# from tqdm import tqdm

API_KEY = st.secrets["general"]["api_key"]

class CombinedMedicalTreeGenerator:
    def __init__(self, file_type: str, disease_context: list,user_input: str):
        self.file_type = file_type
        self.disease_context = disease_context
        self.indication_iterations = 5
        self.technical_iterations = 1
        self.result_iterations = 5
        self.user_input = user_input

        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=API_KEY,
            temperature=0.7
        )
        self.combined_json_filename = "combined_tree.json"
        self.combined_png_filename = "combined_tree"
        self.node_counter = 1

    def generate_alias(self, base_text: str, node_type: str) -> str:
        # This function is kept for deduplication purposes only.
        alias = re.sub(r'[^\w\s]', '', base_text).strip().lower().replace(' ', '_')
        if node_type.lower() in ['question', 'option'] and not alias.startswith(node_type.lower()):
            alias = f"{node_type.lower()}_{alias}"
        return alias

    def extract_section(self, response_content: str) -> str:
        cleaned = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
        cleaned = re.sub(r"```", '', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned

    def parse_indentation_tree(self, tree_str: str) -> list:
        lines = tree_str.splitlines()
        stack = []
        nodes_list = []
        for line in lines:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(' '))
            original_line = line.strip()
            is_list_item = False
            if original_line.startswith("- "):
                is_list_item = True
                original_line = original_line[2:].strip()
            bracket_matches = re.findall(r'\(([^()]*)\)', original_line)
            node_type_extracted = None
            if bracket_matches:
                node_type_extracted = bracket_matches[-1].strip()
                new_text = re.sub(r'\s*\(' + re.escape(node_type_extracted) + r'\)\s*$', '', original_line)
            else:
                new_text = original_line
            if new_text.endswith(":"):
                new_text = new_text[:-1].strip()
            if node_type_extracted is not None:
                node_type = node_type_extracted
            else:
                if not stack:
                    node_type = "root"
                else:
                    if new_text.endswith('?'):
                        node_type = "question"
                    elif is_list_item:
                        node_type = "option"
                    else:
                        node_type = "node"
            while stack and indent <= stack[-1][1]:
                stack.pop()
            if stack:
                parent_node, _ = stack[-1]
                parent_id = parent_node["id"]
                parent_text = parent_node["text"]
            else:
                parent_id = None
                parent_text = None
            node_id = str(self.node_counter)
            self.node_counter += 1
            node = {
                "id": node_id,
                "nodeType": node_type,
                "text": new_text,
                "isLeaf": True,
                "parent": parent_id,
                "parentText": parent_text,
                "childs": []
            }
            nodes_list.append(node)
            if parent_id:
                parent_node["childs"].append(node_id)
                parent_node["isLeaf"] = False
            stack.append((node, indent))
        return nodes_list

    def deduplicate_nodes(self, nodes_list: list) -> tuple:
        node_dict = {node["id"]: node for node in nodes_list}
        memo = {}
        signature_map = {}
        alias_mapping = {}
        def get_signature(node_id):
            if node_id in memo:
                return memo[node_id]
            node = node_dict[node_id]
            child_signatures = tuple(get_signature(child_id) for child_id in node["childs"])
            parent_text = node.get("parentText")
            signature = (node["text"], node["nodeType"], parent_text, child_signatures)
            memo[node_id] = signature
            return signature
        for node_id in node_dict:
            sig = get_signature(node_id)
            if sig not in signature_map:
                signature_map[sig] = node_id
            alias_mapping[node_id] = signature_map[sig]
        for node_id, node in node_dict.items():
            new_childs = []
            for child_id in node["childs"]:
                new_childs.append(alias_mapping[child_id])
            node["childs"] = list(dict.fromkeys(new_childs))
        dedup_node_dict = {}
        for node_id, canonical_id in alias_mapping.items():
            if canonical_id not in dedup_node_dict:
                dedup_node_dict[canonical_id] = node_dict[canonical_id]
        return dedup_node_dict, alias_mapping

    def transform_nodes(self, nodes_dict: dict) -> dict:
        transformed = {}
        for node_id, node in nodes_dict.items():
            new_node = {
                "id": node["id"],
                "nodeType": node["nodeType"],
                "text": node["text"],
                "isLeaf": node["isLeaf"],
                "parent": None,
                "childs": []
            }
            if node["parent"] and node["parent"] in nodes_dict:
                new_node["parent"] = {
                    "id": node["parent"],
                    "text": nodes_dict[node["parent"]]["text"]
                }
            for child_id in node["childs"]:
                if child_id in nodes_dict:
                    child_obj = {
                        "id": child_id,
                        "text": nodes_dict[child_id]["text"]
                    }
                    new_node["childs"].append(child_obj)
            transformed[node_id] = new_node
        return transformed

    def get_node_color(self, node_type: str) -> str:
        color_map = {
            'TYPE_TITLE': 'darkblue',
            'TYPE_TOPIC': 'orange',
            'TYPE_QUESTION': 'lightblue',
            'TYPE_QCM': 'lightgreen',
            'TYPE_QCS': 'lightpink',
            'TYPE_MEASURE': 'yellow',
            'TYPE_DATE': 'violet',
            'TYPE_TEXT': 'tan',
            'TYPE_OPERATION': 'cyan',
            'TYPE_CALCULATION': 'magenta',
            'TYPE_ROOT': 'red'
        }
        return color_map.get(node_type, 'gray')

    def plot_tree(self, nodes, output_filename):
        dot = Digraph(comment='Combined Medical Tree')
        for node_id, node in nodes.items():
            label = node.get('text', node_id)
            fillcolor = self.get_node_color(node.get('nodeType', ''))
            dot.node(node_id, label=label, style='filled', fillcolor=fillcolor)
        for parent_id, node in nodes.items():
            for child in node.get('childs', []):
                if child["id"] in nodes:
                    dot.edge(parent_id, child["id"])
        dot.format = 'png'
        output_path = dot.render(output_filename, view=False)
        print(f"Combined tree plot saved as: {output_path}")

    def generate_indication_tree(self,stream_lit_bar) -> str:
        expanded_prompt = None
        for iteration in range(self.indication_iterations):
            system_instruction = f"""
**goal:**
You are a medical professional. Your task is to generate a structured, hierarchical INDICATION tree for a radiological exam. The tree should clearly document the clinical rationale by including patient details (such as age, sex, and history), the main symptoms prompting the exam, and disease-specific diagnostic questions. This output must be strictly tailored to the file type "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}.

**strictly follow** **User input:**
{self.user_input} **this should be followed strictly**

**return format:**
- Produce exactly one top-level node:
  `INDICATION:` (TYPE_TITLE)
- All subsequent lines must be indented by 4 spaces per level.
- Every node must include its label followed immediately by its nodetype in parentheses. For example:
  - "Patient Information: (TYPE_TOPIC)"
  - "Is the patient experiencing chest pain? (TYPE_QUESTION)"
- Allowed node types include:
  - TYPE_TITLE
  - TYPE_TOPIC
  - TYPE_QUESTION
  - TYPE_QCM - multiple choice answer
  - TYPE_QCS - single choice answer
  - TYPE_MEASURE
  - TYPE_DATE
  - TYPE_TEXT   : free text response
  - TYPE_OPERATION
  - TYPE_CALCULATION


**Additional Details for Special Node Types:**
- TYPE_OPERATION: This node functions as a decision switch using classical Boolean logic. It allows combining conditions using operators such as AND, OR, NOT, >, <, and =. For example, consider the following snippet:

  Symptoms Motivating Examination: (TYPE_TOPIC)
    Combined Respiratory Criteria: (TYPE_OPERATION)
        Is the patient experiencing fever? (TYPE_QUESTION)
            - Yes (TYPE_QCS)
            - No (TYPE_QCS)
        AND
        Is the patient experiencing cough? (TYPE_QUESTION)
            - Yes (TYPE_QCS)
            - No (TYPE_QCS)

  This snippet demonstrates that the branch "Combined Respiratory Criteria" will only trigger further evaluation if both conditions (fever AND cough) are met.
*Note: This example is provided solely for understanding and should not be taken as a literal template; adapt the logical node structure as needed based on the clinical context.*
- TYPE_CALCULATION: This node computes a value based on previously collected measurement responses using basic mathematical operations (addition, subtraction, multiplication, division). For example, it might compute "BMI = weight / (height^2)" when weight and height measurements are provided.

**Example Output Structure (One-Shot):**

INDICATION: (TYPE_TITLE)
    Patient Information: (TYPE_TOPIC)
        Age: (TYPE_QUESTION)
            - Adult (TYPE_QCM)
            - Pediatric (TYPE_QCM)
    Symptoms Motivating Examination: (TYPE_TOPIC)
        Chest Pain: (Question)
            - Sudden severe pain (TYPE_QCM)
            - Dull pressure (TYPE_QCM)
            - Burning sensation (TYPE_QCM)
    ... (other branches)

**warnings:**
- Do not produce more than one top-level "INDICATION:" node; only one is allowed and it must have zero indentation.
- Every branch must include a mandatory "Symptoms Motivating Examination:" section with relevant clinical options.
- Avoid duplicating node names at the same hierarchical level.
- Do not include any extraneous output (such as quotes or additional text) beyond the structured tree.
- **Strictly do not output anything other than the structured output, not even quotes.**

**context dump:**
- The INDICATION tree is specifically for a radiological exam related to "{self.file_type}" and the diseases: {', '.join(self.disease_context)}.
- The structure must encompass both general patient details and a detailed, mandatory "Symptoms Motivating Examination:" section.
- The tree should be designed to support various answer types (MCQ, SCQ, numerical, date, free text) as well as logical and calculation nodes for complex decision-making.
"""
            if iteration == 0:
                user_prompt = f"""
**goal:**
Generate an initial structured INDICATION section for a radiological exam strictly related to "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}.
Focus on listing the major high-level categories and nodes with minimal sub-level detail. Provide an outline that can be expanded in subsequent iterations. A mandatory "Symptoms Motivating Examination:" section must be included.

**return format:**
- A single top-level "INDICATION:" node (with zero indentation) followed by all subordinate nodes indented at 4 spaces per level.
- Every node must include its text and nodetype immediately after (e.g., "Age: (Question)").
- The output should follow the node types and structure outlined in the system instruction.

**warnings:**
- Ensure only one top-level "INDICATION:" node is produced.
- Include the "Symptoms Motivating Examination:" section with relevant clinical options.
- Do not duplicate node names at the same level or include any extra text outside the structured tree.

**context dump:**
- The INDICATION tree is for a radiological exam pertaining to the file type "{self.file_type}" and diseases: {', '.join(self.disease_context)}.
- This initial prompt should produce a broad, high-level outline, establishing general patient details and a basic symptom-related branch that can be further developed in future iterations.
"""
            elif iteration == self.indication_iterations - 1:
                user_prompt = f"""
**goal:**
Refine and fully complete the provided INDICATION section {expanded_prompt} by adding deeper sub-questions to nodes that are still underdeveloped or incomplete. Ensure that every clinically relevant question is addressed and no node remains partially expanded. Focus particularly on expanding disease-specific and symptom-related branches until every clinically relevant question is exhausted.



**return format:**
- Retain the single top-level "INDICATION:" node with all further details indented at 4 spaces per level.
- Every node must continue to follow the format: "Node Text: (Nodetype)".
- Expand branches by including additional sub-nodes such as more detailed symptom queries, logical nodes, or calculation nodes, as appropriate.

**warnings:**
- Do not add any new top-level nodes or duplicate the "INDICATION:" node.
- Avoid unnecessary depth in general nodes (e.g., "Patient Information:") while ensuring that all disease-specific and symptom-related nodes are fully elaborated.
- Ensure that no node names are repeated at the same hierarchical level.

**context dump:**
- This final iteration builds on the existing INDICATION tree for a radiological exam related to "{self.file_type}" and diseases: {', '.join(self.disease_context)}.
- The focus is on finalizing and deepening all disease-specific branches and "Symptoms Motivating Examination:" sections.
- Every node must be fully expanded, ensuring the tree is complete with no unanswered or partially addressed clinical questions.
"""
            else:
                user_prompt = f"""
**goal:**
Refine and expand the existing INDICATION section {expanded_prompt} by increasing the depth of the tree. Add deeper sub-questions and details for disease-specific and symptom-related branches, but do not finalize all nodes. This iteration aims to progressively elaborate the content without completing every branch fully.



**return format:**
- Keep the single top-level "INDICATION:" node with subsequent nodes indented at 4 spaces per level.
- All nodes must include their label followed by the nodetype in parentheses.

**warnings:**
- Only one top-level "INDICATION:" node is allowed; all additional nodes must be indented.
- Do not duplicate node names at the same hierarchical level.
- Ensure the "Symptoms Motivating Examination:" section remains present and is further expanded with clinically relevant details.

**context dump:**
- The INDICATION tree is designed for a radiological exam specifically related to "{self.file_type}" and the diseases: {', '.join(self.disease_context)}.
- This iteration focuses on progressively refining the tree, adding sub-level detail where necessary while leaving room for final completion in later iterations.
"""
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]
            response = self.model.invoke(messages)
            expanded_prompt = self.extract_section(response.content)
            self.current_step+=1
            stream_lit_bar.progress(self.current_step/(self.indication_iterations+self.technical_iterations+self.result_iterations),text=f"INDICATION iteration : {iteration+1} completed")
        print(f"Length of INDICATION tree text: {len(expanded_prompt)}")
        return expanded_prompt

    def generate_technical_tree(self,stream_lit_bar) -> str:
        technical_tree = None
        for iteration in range(self.technical_iterations):
            system_instruction = f"""
**goal:**
You are a medical professional. Your goal is to generate a structured, hierarchical TECHNICAL tree for a radiological exam. This tree should detail the technical parameters and protocols used during imaging—such as the use of contrast injections, imaging sequences (e.g., T1, T2, FLAIR, angiographic sequences), and other modality-specific settings. This output must be strictly tailored to the file type "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}. The TECHNICAL tree will typically follow the INDICATION tree {{expanded_prompt}} for context, but it should not duplicate information from the INDICATION or RESULT trees.

**strictly follow** **User input:**
{self.user_input} this should be followed strictly

**return format:**
- Produce exactly one top-level node:
  `TECHNICAL:` (TYPE_TITLE)
- All subsequent lines must be indented by 4 spaces per level.
- Each node must include its label followed immediately by its nodetype in parentheses. For example:
  - "Injection Protocol: (TYPE_TOPIC)"
  - "Is contrast injection used? (TYPE_QUESTION)"
  - "Yes (TYPE_QCS)"
- Allowed node types include:
  - TYPE_TITLE
  - TYPE_TOPIC
  - TYPE_QUESTION
  - TYPE_QCM - multiple choice answer
  - TYPE_QCS - single choice answer
  - TYPE_MEASURE
  - TYPE_DATE
  - TYPE_TEXT   : free text response
  - TYPE_OPERATION
  - TYPE_CALCULATION



**Example Output Structure (One-Shot):**

TECHNICAL: (Title)
    Injection Protocol: (Topic)
        Is contrast injection used? (TYPE_QUESTION)
            - Yes (TYPE_QCS)
            - No (TYPE_QCS)
    Sequences: (Topic)
        Ax T1: (TYPE_QCS)
        Ax T2: (TYPE_QCS)
        Ax FLAIR: (TYPE_QCS)
        3D T1 IR: (TYPE_QCS)
        ...
    Additional Parameters: (Topic)
        Any specific coil used? (TYPE_QUESTION)
            - Head coil (TYPE_QCS)
            - Neck coil (TYPE_QCS)
            - Multichannel coil (TYPE_QCS)

**warnings:**
- Do not produce more than one top-level "TECHNICAL:" node; only one is allowed and it must have zero indentation.
- Group technical details logically (e.g., injection protocol, sequences, additional parameters).
- Avoid duplicating node names at the same hierarchical level.
- Do not include any extraneous output (such as quotes or additional text) beyond the structured tree.

**context dump:**
- The TECHNICAL tree is meant to capture all relevant imaging protocols for a radiological exam of type "{self.file_type}" in the context of diseases: {', '.join(self.disease_context)}.
- Nodes can represent whether contrast was used, what sequences or series were acquired, and any special imaging parameters (e.g., coil types, slice thickness).
- This structure ensures that the imaging procedure is clearly documented, providing consistency for subsequent interpretation and correlation with the INDICATION and RESULT trees.
"""
            user_prompt = f"""
**goal:**
Generate a single structured and deep TECHNICAL section for a radiological exam strictly based on the imaging protocols used. This output must be tailored to the file type "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}. The tree should cover key technical aspects such as contrast injection usage, imaging sequences, and any additional parameters relevant to the modality.



**return format:**
- A single top-level "TECHNICAL:" node (with zero indentation) followed by all subordinate nodes indented at 4 spaces per level.
- Each node must include its text and nodetype immediately after (e.g., "Injection Protocol: (Topic)").
- The output should follow the node types and structure outlined in the system instruction.

**warnings:**
- Ensure only one top-level "TECHNICAL:" node is produced.
- Avoid duplicating node names at the same level.
- Do not include any extra text outside the structured tree.

**context dump:**
- The TECHNICAL tree references the specific imaging protocols after the INDICATION tree, so it should logically reflect the sequences and parameters necessary for the file type "{self.file_type}" and diseases: {', '.join(self.disease_context)}.
- This prompt requires a comprehensive but not overly complex structure, ensuring major parameters (e.g., contrast usage, sequence list, coil or scanning parameters) are included without redundancy.
"""
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]
            response = self.model.invoke(messages)
            technical_tree = self.extract_section(response.content)
            self.current_step+=1
            stream_lit_bar.progress(self.current_step/(self.indication_iterations+self.technical_iterations+self.result_iterations),text=f"TECHNIQUE iteration : {iteration+1} completed")
        print(f"Length of TECHNICAL tree text: {len(technical_tree)}")
        return technical_tree

    def generate_result_tree(self, indication_tree_text: str, technical_tree_text: str,stream_lit_bar) -> str:
        result = None
        for iteration in range(self.result_iterations):
            if iteration == 0:
                user_prompt = f"""
**goal:**
Generate an initial structured RESULT section for a radiological exam based on final imaging observations.
Tailor the output to "{self.file_type}" and diseases: {', '.join(self.disease_context)}.
List major anatomical categories (e.g., pleura, parenchyma, mediastinum, bones, devices) with minimal detail.

"""

            elif iteration == self.result_iterations - 1:
                user_prompt = f"""
**goal:**
Refine and fully complete the provided RESULT section {result} by adding deeper sub-questions or nodes for each anatomical category. Focus on detailing any abnormalities (e.g., describing size, extent, severity, specific locations) and including measurement, logical, or calculation nodes as necessary. Ensure that every node is properly generated without any cut-offs.

**return format:**
- Retain the single top-level "RESULT:" node with all further details indented at 4 spaces per level.
- Every node must continue to follow the format: "Node Text: (Nodetype)".
- Expand branches by including additional sub-nodes such as more detailed abnormality classifications, measurement nodes, logical nodes, or calculation nodes.



"""
            else:
                user_prompt = f"""
**goal:**
Refine and expand the existing RESULT section {result} by adding deeper sub-questions and details where clinically appropriate. Emphasize further elaboration of abnormal findings while maintaining the overall structure.

**return format:**
- Maintain a single top-level "RESULT:" node with subsequent nodes indented at 4 spaces per level.
- All nodes must include their label followed by the nodetype in parentheses.
"""
            system_instruction = f"""
**goal:**
You are a medical professional. Your goal is to generate a structured, hierarchical RESULT tree for a radiological exam. This tree should detail the radiological findings in a systematic manner, capturing observations about anatomical structures (e.g., pleura, parenchyma, mediastinum, bone structures, devices) and any detected abnormalities (e.g., effusions, nodules, calcifications). This output must be strictly tailored to the file type "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}. The RESULT tree will typically follow the completion of an INDICATION tree:
{indication_tree_text}
and a TECHNICAL tree:
{technical_tree_text}
both of which may be referenced for context but should not be duplicated here.

**strictly follow** **User input:**
{self.user_input} this should be followed strictly

**return format:**
- Produce exactly one top-level node:
  `RESULT:` (TYPE_TITLE)
- All subsequent lines must be indented by 4 spaces per level.
- Each node must include its label followed immediately by its nodetype in parentheses. For example:
  - "Pleura: (TYPE_TOPIC)"
  - "Is there a pleural effusion? (TYPE_QUESTION)"
  - "None (TYPE_QCS)"
- Allowed node types include:
  - TYPE_TITLE
  - TYPE_TOPIC
  - TYPE_QUESTION
  - TYPE_QCM - multiple choice answer
  - TYPE_QCS - single choice answer
  - TYPE_MEASURE
  - TYPE_DATE
  - TYPE_TEXT   : free text response
  - TYPE_OPERATION
  - TYPE_CALCULATION


**Additional Details for Special Node Types:**
- Logical node: This node functions as a decision switch using classical Boolean logic. It allows combining conditions using operators such as AND, OR, NOT, >, <, and =. For example, consider the following snippet:

  Symptoms Motivating Examination: (TYPE_TOPIC)
    Combined Respiratory Criteria: (TYPE_OPERATION)
        Is the patient experiencing fever? (TYPE_QUESTION)
            - Yes (TYPE_QCS)
            - No (TYPE_QCS)
        AND
        Is the patient experiencing cough? (TYPE_QUESTION)
            - Yes (TYPE_QCS)
            - No (TYPE_QCS)

  This snippet demonstrates that the branch "Combined Respiratory Criteria" will only trigger further evaluation if both conditions (fever AND cough) are met.
*Note: This example is provided solely for understanding and should not be taken as a literal template; adapt the logical node structure as needed based on the clinical context.*
- TYPE_CALCULATION: This node computes a value based on previously collected measurement responses using basic mathematical operations (addition, subtraction, multiplication, division). For example, it might compute "BMI = weight / (height^2)" when weight and height measurements are provided.

**Example Output Structure (One-Shot):**
RESULT: (TYPE_TITLE)
  Pleura: (TYPE_TOPIC)
      Is there a pleural effusion? (TYPE_QUESTION)
          - None (TYPE_QCS)
          - Mild (TYPE_QCS)
          - Moderate (TYPE_QCS)
          - Large (TYPE_QCS)
      Is there a pneumothorax? (Question)
          - Yes (TYPE_QCS)
          - No (TYPE_QCS)
  Parenchyma: (Topic)
      Presence of parenchymal abnormality: (TYPE_QUESTION)
          - Mass (TYPE_QCM)
          - Nodule (TYPE_QCM)
          - Consolidation (TYPE_QCM)
      Appearance of parenchyma: (Topic)
          Are there interstitial changes? (TYPE_QUESTION)
              - Yes (TYPE_QCS)
              - No (TYPE_QCS)
  Mediastinum: (Topic)
      Any mediastinal enlargement? (TYPE_QUESTION)
          - Yes (TYPE_QCS)
          - No (TYPE_QCS)
      Calcification: (Topic)
          Which element is calcified? (TYPE_QUESTION)
              - Aorta (TYPE_QCM)
              - Lymph node (TYPE_QCM)
              - Valve (TYPE_QCM)
  ...

**warnings:**
- Do not produce more than one top-level "RESULT:" node; only one is allowed and it must have zero indentation.
- Ensure findings are grouped logically (e.g., pleural, parenchymal, mediastinal, skeletal, etc.) and further broken down by abnormal or normal findings.
- Avoid duplicating node names at the same hierarchical level.
- Do not include any extraneous output (such as quotes or additional text) beyond the structured tree.
- **Strictly do not output anything other than the structured output, not even quotes.**

**context dump:**
- The RESULT tree should capture the final imaging observations from a radiological exam, potentially referencing information from the previously filled INDICATION tree and TECHNICAL tree.
- Nodes can represent normal or abnormal findings, sub-classifications of abnormalities, measurement details, or additional descriptive text where clinically relevant.
- This structure is designed to accommodate detailed reporting of radiological findings, ensuring clarity and consistency in how results are documented.
"""
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]
            response = self.model.invoke(messages)
            result = self.extract_section(response.content)
            self.current_step+=1
            stream_lit_bar.progress(self.current_step/(self.indication_iterations+self.technical_iterations+self.result_iterations),text=f"RESULT iteration : {iteration+1} completed")
        print(f"Length of RESULT tree text: {len(result)}")
        return result

    def combine_trees(self, indication_nodes: list, technical_nodes: list, result_nodes: list) -> dict:
        def get_root(nodes):
            for node in nodes:
                if node.get("parent") is None:
                    return node
            return None

        indication_root = get_root(indication_nodes)
        technical_root = get_root(technical_nodes)
        result_root = get_root(result_nodes)
        new_root_id = str(self.node_counter)
        self.node_counter += 1
        new_root = {
            "id": new_root_id,
            "nodeType": "TYPE_ROOT",
            "text": self.file_type,
            "isLeaf": False,
            "parent": None,
            "parentText": None,
            "childs": []
        }
        if indication_root:
            indication_root["parent"] = new_root_id
            indication_root["parentText"] = self.file_type
            new_root["childs"].append(indication_root["id"])
        if technical_root:
            technical_root["parent"] = new_root_id
            technical_root["parentText"] = self.file_type
            new_root["childs"].append(technical_root["id"])
        if result_root:
            result_root["parent"] = new_root_id
            result_root["parentText"] = self.file_type
            new_root["childs"].append(result_root["id"])
        combined_nodes = [new_root] + indication_nodes + technical_nodes + result_nodes
        dedup_nodes, _ = self.deduplicate_nodes(combined_nodes)
        return dedup_nodes

    def run(self,stream_lit_bar,stream_lit_text):
        self.current_step = 0
        stream_lit_text.text("Generating INDICATION tree...")
        stream_lit_bar.progress(self.current_step/(self.indication_iterations+self.technical_iterations+self.result_iterations),"Starting Indication tree generation")
        indication_text = self.generate_indication_tree(stream_lit_bar=stream_lit_bar)
        stream_lit_text.text("Successfully generated INDICATION tree. Generating TECHNICAL tree...")
        stream_lit_bar.progress(self.current_step/(self.indication_iterations+self.technical_iterations+self.result_iterations),"Starting Technical tree generation")
        technical_text = self.generate_technical_tree(stream_lit_bar=stream_lit_bar)
        stream_lit_text.text("Successfully generated TECHNIQUE tree. Generating RESULT tree...")
        stream_lit_bar.progress(self.current_step/(self.indication_iterations+self.technical_iterations+self.result_iterations),"Starting Result tree generation")
        result_text = self.generate_result_tree(indication_text, technical_text,stream_lit_bar=stream_lit_bar)
        indication_nodes = self.parse_indentation_tree(indication_text)
        technical_nodes = self.parse_indentation_tree(technical_text)
        result_nodes = self.parse_indentation_tree(result_text)

        indication_dedup, _ = self.deduplicate_nodes(indication_nodes)
        technical_dedup, _ = self.deduplicate_nodes(technical_nodes)
        result_dedup, _ = self.deduplicate_nodes(result_nodes)
        print(f"length of indication tree :{len(indication_dedup)}")
        print(f"length of technical tree :{len(technical_dedup)}")
        print(f"length of result tree :{len(result_dedup)}")
        print(f"sum: {len(indication_dedup)+len(technical_dedup)+len(result_dedup)}")
        combined_nodes = self.combine_trees(list(indication_dedup.values()),
                                            list(technical_dedup.values()),
                                            list(result_dedup.values()))
        print(f"Length of combined tree: {len(combined_nodes)}")
        transformed_nodes = self.transform_nodes(combined_nodes)
        stream_lit_text.text("Successfully generated and processed tree")
        print(f"Returning the tree")
        return list(transformed_nodes.values())
                    
        # with open(self.combined_json_filename, "w") as f:
        #     json.dump(list(transformed_nodes.values()), f, indent=2)
        # print(f"Combined JSON saved to {self.combined_json_filename}")

        # # self.plot_tree(transformed_nodes, self.combined_png_filename)
        # print(f"Length of combined tree: {len(transformed_nodes)}")
        # stream_lit_text.text("Completed and processed generated tree")