import re
# import os
# import json
from graphviz import Digraph
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from deep_translator import GoogleTranslator
# from tqdm import tqdm

API_KEY = st.secrets["general"]["api_key"]

class CombinedMedicalTreeGenerator:
    def __init__(self, file_type: str, disease_context: list, user_input: str):
        self.file_type = file_type
        self.disease_context = disease_context
        self.user_input = user_input
        # Fixed iteration counts: 5 for INDICATION and RESULT, 1 for TECHNICAL
        self.indication_iterations = 5
        self.technical_iterations = 2
        self.result_iterations = 5

        # self.model_eval = ChatGroq(
        #     model_name="llama-3.3-70b-versatile",
        #     api_key="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",
        #     temperature=0.7
        # )
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=API_KEY,
            temperature=0.7
        )
        self.model_eval = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=API_KEY,
            temperature=0.7
        )
        # Final output filenames
        self.combined_json_filename = f"{self.file_type}_combined_tree.json"
        self.combined_png_filename = "combined_tree"
        # Counter used for JSON conversion unique IDs
        self.node_counter = 1

    def extract_section(self, response_content: str) -> str:
        # Remove any <think> sections, triple backticks, and normalize blank lines.
        cleaned = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
        cleaned = re.sub(r"```", '', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned

    # ---------------- Evaluation for Each Tree ----------------

    def evaluate_tree_indication(self, tree_text: str) -> str:
        evaluation_instruction = f"""
You are an expert in clinical decision trees. Please evaluate the following INDICATION tree using the criteria below and provide your feedback in a clear, structured format with actionable recommendations.

1. GOAL
   - Assess the subtree for clinical accuracy, logical flow, and efficient structuring.
   - Ensure it faithfully implements the TYPE_* schema and any user‑specified requirements.

2. EVALUATION CRITERIA

    is the below Structure followed strictly by the tree? :
      1. SECTIONS
        create each major section as:
        - "<Section Title>" (TYPE_TITLE)

      2. QUESTIONS & OPTIONS
        Under each TYPE_TITLE, build a sequence of questions and answer options:
        a. Ask a clinical question:
            - "<Question text>?" (TYPE_QUESTION)
        b. List its answer choices:
            - "<Option A>" (TYPE_QCS) or (TYPE_QCM)
            - "<Option B>" (TYPE_QCS) or (TYPE_QCM)
            …
        c. If any option requires further detail, branch again with:
            - "<Follow-up question>?" (TYPE_QUESTION)
              * Under that, list its choices or fields:
                - "<Field Name>" (TYPE_QCS/QCM)
                - "<Measure Name>" (TYPE_MEASURE) (unit)
                - "<Text Field>" (TYPE_TEXT)


      4. PARENT QUESTION FOR MEASURE/TEXT
         **Strictly Whenever you include a TYPE_MEASURE or TYPE_TEXT leaf, first insert its parent question (TYPE_QUESTION) with the same label for each one of it**.
      - Every node must be formatted as:
          Node Text (Nodetype)
          If the nodetype is TYPE_MEASURE, include the unit in parentheses (e.g., `(years)`, `(mm)`).
           eg : Duration: (TYPE_MEASURE) (days)
      5. Phrase in the leaf nodes :
        For every leaf node (i.e. a node of type TYPE_QCM, TYPE_QCS, TYPE_MEASURE, TYPE_DATE or TYPE_TEXT with no children), append a pipe and a quoted, radiology-style summary phrase or sentence immediately after its label,
        for example format : Duration (TYPE_MEASURE) (days)  |  "Antécédent de la durée de la condition en jours."
                            Mild (TYPE_QCS) |  "Antécédent de fièvre légère."

       **Example** :
            INDICATION (TYPE_TITLE)
                Quel est le motif principal de l’examen ? (TYPE_TITLE)
                    Douleur (TYPE_QCM)  |  "Douleur signalée."
                        Où se situe la douleur ? (TYPE_QUESTION)
                            Région anatomique (TYPE_TEXT)  |  "Localisation de la douleur spécifiée."
                        Depuis combien de temps la douleur persiste-t-elle ? (TYPE_QUESTION)
                            Durée de la douleur (TYPE_MEASURE) (jours)  |  "de douleur"
                    Traumatisme (TYPE_QCM)  |  "Traumatisme suspecté."
                        Type de traumatisme suspecté ? (TYPE_QUESTION)
                            Contusion (TYPE_QCS)          |  "Contusion suspectée."
                            Fracture (TYPE_QCS)           |  "Fracture suspectée."
                            Autre traumatisme (TYPE_QCS)  |  "Autre traumatisme signalé."
                                Précisez le traumatisme ? (TYPE_QUESTION)
                                    Détails du traumatisme (TYPE_TEXT)  |  "Description du traumatisme fournie."
                    Infection (TYPE_QCM)  |  "Infection suspectée."
                        Quels symptômes infectieux ? (TYPE_QUESTION)
                            Fièvre (TYPE_QCS)                    |  "Fièvre signalée."
                            Écoulement purulent (TYPE_QCS)       |  "Écoulement purulent signalé."
                            Autre symptôme infectieux (TYPE_QCS) |  "Autre symptôme infectieux signalé."
                                Précisez le symptôme ? (TYPE_QUESTION)
                                    Détails du symptôme (TYPE_TEXT)  |  "Description du symptôme infectieux fournie."
                    Bilan de suivi (TYPE_QCM)  |  "Examen de suivi demandé."
                        Intervalle depuis le dernier examen ? (TYPE_QUESTION)
                            Intervalle de suivi (TYPE_MEASURE) (mois)  |  " depuis le dernier examen"
                    Autre indication (TYPE_QCM)  |  "Autre indication signalée."
                        Précisez l’indication ? (TYPE_QUESTION)
                            Détails de l’indication (TYPE_TEXT)  |  "Détails de l’indication fournis."

   1. **Clarity & Clinical Relevance**
      - Are questions phrased clearly, in the target language/the {self.disease_context} language (per user’s locale)?
      - Do answer options cover the clinically meaningful possibilities?
      - Are follow‑up questions logically branched under the appropriate parent node?




   4. **Efficiency & Node Management**
      - Are there no more than 5 immediate children per node?
      - If more than five options exist, are the most common ones listed first with an “Other” node?
      - **Strictly ensure that related yes/no sequences consolidated into multi‑option questions where it makes sense?**
   5. **Mixed Answer‑Type Handling**
      - When a question logically requires both multiple (TYPE_QCM) and single (TYPE_QCS) choice answers, are both types present?
      - Are any TYPE_MEASURE or TYPE_TEXT options improperly mixed into TYPE_QCS and TYPE_QCM nodes?  If so, should be separated into their own questions?
   6. **Actionable Recommendations**
      - For each issue you identify, propose a concrete fix: merge nodes, correct a node type, add a missing branch, simplify wording, etc.

3. RETURN FORMAT
   Structure your response as three sections:
   **A. Summary of Strengths**
     – List up to three aspects the subtree does well.
   **B. Issues & Recommendations**
     – For each criterion above, name any problems (or “None”) and immediately follow with “Recommendation: …”
   **C. Additional Notes**
     – Any miscellaneous observations (e.g., locale/language mismatches).

4. WARNINGS
   - Do **not** echo the entire tree.
   - Output only your evaluation sections.
   - If the user supplied extra context (e.g., “user_input”), verify it’s honored and note any discrepancies.


7. user input {self.user_input} if provided ensure that it is included in the evaluation ensuring it is folllowed.


The INDICATION tree to evaluate:
{tree_text}

Please provide your evaluation in a clear, structured format with your recommendations.
"""
        user_prompt = "Please provide your evaluation in a clear, structured format with your recommendations."
        messages = [SystemMessage(content=evaluation_instruction), HumanMessage(content=user_prompt)]
        response = self.model_eval.invoke(messages)
        evaluation_feedback = self.extract_section(response.content.strip())
        return evaluation_feedback

    def evaluate_tree_technical(self, tree_text: str) -> str:
        evaluation_instruction = f"""
You are an expert in radiological technical protocols. Please evaluate the following TECHNICAL tree using the criteria below and provide your feedback with actionable recommendations.

1. GOAL
   - Assess the subtree for clinical accuracy, logical flow, and efficient structuring.
   - Ensure it faithfully implements the TYPE_* schema and any user‑specified requirements.

2. EVALUATION CRITERIA
    is the below Structure followed strictly by the tree? :
      1. SECTIONS
        create each major section as:
        - "<Section Title>" (TYPE_TITLE)

      2. QUESTIONS & OPTIONS
        Under each TYPE_TITLE, build a sequence of questions and answer options:
        a. Ask a clinical question:
            - "<Question text>?" (TYPE_QUESTION)
        b. List its answer choices:
            - "<Option A>" (TYPE_QCS) or (TYPE_QCM)
            - "<Option B>" (TYPE_QCS) or (TYPE_QCM)
            …
        c. If any option requires further detail, branch again with:
            - "<Follow-up question>?" (TYPE_QUESTION)
              * Under that, list its choices or fields:
                - "<Field Name>" (TYPE_QCS/QCM)
                - "<Measure Name>" (TYPE_MEASURE) (unit)
                - "<Text Field>" (TYPE_TEXT)


      4. PARENT QUESTION FOR MEASURE/TEXT
         **Strictly Whenever you include a TYPE_MEASURE or TYPE_TEXT leaf, first insert its parent question (TYPE_QUESTION) with the same label for each one of it**.
      - Every node must be formatted as:
          Node Text (Nodetype)
          If the nodetype is TYPE_MEASURE, include the unit in parentheses (e.g., `(years)`, `(mm)`).
           eg : Duration: (TYPE_MEASURE) (days)
      5. Phrase in the leaf nodes :
        For every leaf node (i.e. a node of type TYPE_QCM, TYPE_QCS, TYPE_MEASURE, TYPE_DATE or TYPE_TEXT with no children), append a pipe and a quoted, radiology-style summary phrase or sentence immediately after its label,
        for example format : Duration (TYPE_MEASURE) (days)  |  "Antécédent de la durée de la condition en jours."
                            Mild (TYPE_QCS) |  "Antécédent de fièvre légère."

        **Example** :
       TECHNIQUE (TYPE_TITLE)
          Détails de la technique (TYPE_TITLE)
              Quelle modalité d’imagerie a été utilisée ? (TYPE_QUESTION)
                  Échographie (TYPE_QCS)  |  "Échographie réalisée."
                      Quel transducteur a été utilisé ? (TYPE_QUESTION)
                          Transducteur (TYPE_TEXT)  |  "Transducteur spécifié."
                      Quelle fréquence de sonde ? (TYPE_QUESTION)
                          Fréquence (TYPE_MEASURE) (MHz)  |  "MHz de fréquence de sonde."
                      Doppler appliqué ? (TYPE_QUESTION)
                          Oui (TYPE_QCS)  |  "Doppler appliqué."
                          Non (TYPE_QCS)  |  "Doppler non appliqué."
                  Scanner (TYPE_QCS)  |  "Scanner réalisé."
                      Quel voltage a été utilisé ? (TYPE_QUESTION)
                          Voltage (TYPE_MEASURE) (kV)  |  " kV de tension."
                      Quel débit a été utilisé ? (TYPE_QUESTION)
                          Débit (TYPE_MEASURE) (mAs)  |  " mAs de débit."
                      Contraste iodé injecté ? (TYPE_QUESTION)
                          Oui (TYPE_QCS)  |  "Contraste iodé injecté."
                              Quel volume de contraste ? (TYPE_QUESTION)
                                  Volume (TYPE_MEASURE) (mL)  |  " mL de contraste injecté."
                          Non (TYPE_QCS)  |  "Pas de contraste iodé injecté."
                  IRM (TYPE_QCS)  |  "IRM réalisée."
                      Séquences utilisées ? (TYPE_QUESTION)
                          Séquences (TYPE_TEXT)  |  "Séquences spécifiées."
                  Autre modalité (TYPE_QCS)  |  "Autre modalité précisée."
                      Précisez la modalité ? (TYPE_QUESTION)
                          Modalité (TYPE_TEXT)  |  "Modalité précisée."



   1. **Clarity & Clinical Relevance**
      - Are questions phrased clearly, in the target language/the {self.disease_context} language (per user’s locale)?
      - Do answer options cover the clinically meaningful possibilities?
      - Are follow‑up questions logically branched under the appropriate parent node?


   4. **Efficiency & Node Management**
      - Are there no more than 5 immediate children per node?
      - If more than five options exist, are the most common ones listed first with an “Other” node?
      - **Strictly ensure that related yes/no sequences consolidated into multi‑option questions where it makes sense?**
   5. **Mixed Answer‑Type Handling**
      - When a question logically requires both multiple (TYPE_QCM) and single (TYPE_QCS) choice answers, are both types present?
      - Are any TYPE_MEASURE or TYPE_TEXT options improperly mixed into TYPE_QCS and TYPE_QCM nodes?  If so, should be separated into their own questions?
   6. **Actionable Recommendations**
      - For each issue you identify, propose a concrete fix: merge nodes, correct a node type, add a missing branch, simplify wording, etc.

3. RETURN FORMAT
   Structure your response as three sections:
   **A. Summary of Strengths**
     – List up to three aspects the subtree does well.
   **B. Issues & Recommendations**
     – For each criterion above, name any problems (or “None”) and immediately follow with “Recommendation: …”
   **C. Additional Notes**
     – Any miscellaneous observations (e.g., locale/language mismatches).

4. WARNINGS
   - Do **not** echo the entire tree.
   - Output only your evaluation sections.
   - If the user supplied extra context (e.g., “user_input”), verify it’s honored and note any discrepancies.


The TECHNICAL tree to evaluate:
{tree_text}

Please provide your evaluation in a clear, structured format with your recommendations.
"""
        user_prompt = "Please provide your evaluation in a clear, structured format with your recommendations."
        messages = [SystemMessage(content=evaluation_instruction), HumanMessage(content=user_prompt)]
        response = self.model_eval.invoke(messages)
        evaluation_feedback = self.extract_section(response.content.strip())
        return evaluation_feedback

    def evaluate_tree_result(self, tree_text: str) -> str:
        evaluation_instruction = f"""
You are an expert in radiological reporting. Please evaluate the following RESULT tree using the criteria below and provide your feedback in a clear, structured format with actionable recommendations.
Criteria:

1. **Clarity and Quality of Observations:**
   - is the content generated in french except for the node type and unit which are in the brackets.?
   - **Strictly when options are branched again a relavent question(TYPE_QUESTION) should be asked and then options then again question and the cycle continues.?**
   - ensure that the question is of relevant detail and meaningful.
   - Are the anatomical categories (e.g., pleura, parenchyma, mediastinum, bones, devices) clearly defined?
   - Are the clinical observations and measurements concise and meaningful?

2. **Correctness and Consistency of Node Types:**
   - Are the correct node types used (e.g., TYPE_TITLE, TYPE_TOPIC, TYPE_QUESTION, TYPE_QCM, TYPE_QCS, TYPE_MEASURE, TYPE_DATE, TYPE_TEXT)?
   - Is the formatting consistent throughout the tree?
   - is TYPE_QCM and TYPE_QCS are used logically, and TYPE_MEASURE is applied where applicable?
   - in the options is TYPE_TEXT and TYPE_MEASURE is mixed up with TYPE_QCM and TYPE_QCS? if yes then it should be asked as a separate question.
   - in the place where measurement is needed TYPE_MEASURE can be used (eg: length,breadth,volume etc can be asked as options of same question with TYPE_MEASURE)

2. EVALUATION CRITERIA

    is the below Structure followed strictly by the tree? :
      1. SECTIONS
        create each major section as:
        - "<Section Title>" (TYPE_TITLE)

      2. QUESTIONS & OPTIONS
        Under each TYPE_TITLE, build a sequence of questions and answer options:
        a. Ask a clinical question:
            - "<Question text>?" (TYPE_QUESTION)
        b. List its answer choices:
            - "<Option A>" (TYPE_QCS) or (TYPE_QCM)
            - "<Option B>" (TYPE_QCS) or (TYPE_QCM)
            …
        c. If any option requires further detail, branch again with:
            - "<Follow-up question>?" (TYPE_QUESTION)
              * Under that, list its choices or fields:
                - "<Field Name>" (TYPE_QCS/QCM)
                - "<Measure Name>" (TYPE_MEASURE) (unit)
                - "<Text Field>" (TYPE_TEXT)


      4. PARENT QUESTION FOR MEASURE/TEXT
        **Strictly Whenever you include a TYPE_MEASURE or TYPE_TEXT leaf, first insert its parent question (TYPE_QUESTION) with the same label for each one of it**.
      - Every node must be formatted as:
          Node Text: (Nodetype)
          If the nodetype is TYPE_MEASURE, include the unit in parentheses (e.g., `(years)`, `(mm)`).
           eg : Duration: (TYPE_MEASURE) (days)
      5. Phrase in the leaf nodes :
        For every leaf node (i.e. a node of type TYPE_QCM, TYPE_QCS, TYPE_MEASURE, TYPE_DATE or TYPE_TEXT with no children), append a pipe and a quoted, radiology-style summary phrase or summary immediately after its label,
        for example format : Duration (TYPE_MEASURE) (days)  |  "Antécédent de la durée de la condition en jours."
                            Mild (TYPE_QCS) |  "Antécédent de fièvre légère."

      **Example** :
          RÉSULTAT (TYPE_TITLE)
              Morphologie et dimensions (TYPE_TITLE)
                  Quelle est la taille du lobe droit ? (TYPE_QUESTION)
                      Taille du lobe droit (TYPE_MEASURE) (mm)  |  " de lobe droit"
                  Quelle est la taille du lobe gauche ? (TYPE_QUESTION)
                      Taille du lobe gauche (TYPE_MEASURE) (mm)  |  "de lobe gauche"
                  Quelle est l’épaisseur de l’isthme ? (TYPE_QUESTION)
                      Épaisseur de l’isthme (TYPE_MEASURE) (mm)  |  " d’isthme"
                  Quel est le volume global de la thyroïde ? (TYPE_QUESTION)
                      Volume global de la thyroïde (TYPE_MEASURE) (mL)  |  "de volume thyroïdien"
              Présence d’anomalies morphologiques (TYPE_TITLE)
                  Oui (TYPE_QCS)  |  "Anomalie morphologique présente"
                  Non (TYPE_QCS)  |  "Pas d’anomalie morphologique"
                  Veuillez décrire l’anomalie morphologique (TYPE_QUESTION)
                      Description de l'anomalie (TYPE_TEXT)  |  "Anomalie morphologique décrite"
                  Quel type d’anomalie morphologique ? (TYPE_QUESTION)
                      Agénésie (TYPE_QCS)    |  "Agénésie"
                      Hémiagénésie (TYPE_QCS)   |  "Hémiagénésie"
                      Lobe pyramidal proéminent (TYPE_QCS)  |  "Lobe pyramidal proéminent"
                      Thyroïde ectopique (TYPE_QCS)  |  "Thyroïde ectopique"
                      Autre (TYPE_QCS)   |  "Autre anomalie morphologique"
                      Précisez l’anomalie morphologique (TYPE_QUESTION)
                          Précisez l’anomalie (TYPE_TEXT)    |  "Détail de l’anomalie fourni"
              Échostructure du parenchyme thyroïdien (TYPE_TITLE)
                  Homogène (TYPE_QCS)  |  "Parenchyme homogène"
                  Hétérogène (TYPE_QCS)   |  "Parenchyme hétérogène"
                  Veuillez décrire l’hétérogénéité (TYPE_QUESTION)
                      Description de l'hétérogénéité (TYPE_TEXT)  |  "Hétérogénéité décrite"
                  Quel type d’hétérogénéité ? (TYPE_QUESTION)
                      Focalisée (TYPE_QCS)   |  "Hétérogénéité focalisée"
                      Diffuse (TYPE_QCS)   |  "Hétérogénéité diffuse"
                      Mixte (TYPE_QCS)    |  "Hétérogénéité mixte"



   1. **Clarity & Clinical Relevance**
      - Are questions phrased clearly, in the target language/the {self.disease_context} language (per user’s locale)?
      - Do answer options cover the clinically meaningful possibilities?
      - Are follow‑up questions logically branched under the appropriate parent node?

   4. **Efficiency & Node Management**
      - Are there no more than 5 immediate children per node?
      - If more than five options exist, are the most common ones listed first with an “Other” node?
      - **Strictly ensure that related yes/no sequences consolidated into multi‑option questions where it makes sense?**
   5. **Mixed Answer‑Type Handling**
      - When a question logically requires both multiple (TYPE_QCM) and single (TYPE_QCS) choice answers, are both types present?
      - Are any TYPE_MEASURE or TYPE_TEXT options improperly mixed into TYPE_QCS and TYPE_QCM nodes?  If so, should be separated into their own questions?
   6. **Actionable Recommendations**
      - For each issue you identify, propose a concrete fix: merge nodes, correct a node type, add a missing branch, simplify wording, etc.

3. RETURN FORMAT
   Structure your response as three sections:
   **A. Summary of Strengths**
     – List up to three aspects the subtree does well.
   **B. Issues & Recommendations**
     – For each criterion above, name any problems (or “None”) and immediately follow with “Recommendation: …”
   **C. Additional Notes**
     – Any miscellaneous observations (e.g., locale/language mismatches).

4. WARNINGS
   - Do **not** echo the entire tree.
   - Output only your evaluation sections.
   - If the user supplied extra context (e.g., “user_input”), verify it’s honored and note any discrepancies.

7. user input {self.user_input} if provided ensure that it is included in the evaluation ensuring it is folllowed.


The RÉSULTAT tree to evaluate:
{tree_text}

Please provide your evaluation in a clear, structured format with your recommendations.
"""
        user_prompt = "Please provide your evaluation in a clear, structured format with your recommendations."
        messages = [SystemMessage(content=evaluation_instruction), HumanMessage(content=user_prompt)]
        response = self.model_eval.invoke(messages)
        evaluation_feedback = self.extract_section(response.content.strip())
        return evaluation_feedback

    # ---------------- Tree Generation Methods ----------------

    def generate_indication_tree(self) -> str:
        expanded_prompt = None
        evaluation_feedback = ""
        for iteration in range(self.indication_iterations):
            print(f"Generating INDICATION tree: iteration {iteration+1}...")
            system_instruction = f"""
**Goal:**
You are a medical professional tasked with generating a structured, hierarchical INDICATION tree for a radiological exam. The tree must document the clinical rationale by including patient details (such as age, sex, and history), the primary symptoms prompting the exam, and disease-specific diagnostic questions. This output is strictly tailored to the file type "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}.

**User input:**
{self.user_input} **is the user recommandation and this should be followed strictly**

**Strict Guidelines:**
1. **Non-Empty, Contextual Nodes:**
   - Every node must contain clear, context-specific text. No node should be empty.

2. **Quality Clinical Questions & Answer Options:**
   - is the content generated in french except for the node type and unit which are in the brackets.?
   - **Strictly when options are branched again a relavent question(TYPE_QUESTION) should be asked and then options then again question and the cycle continues.?**
   - Design questions and answer options that are logically sound and clinically meaningful.
   - Prioritize quality over quantity: craft questions that combine related clinical queries to reduce the total number of nodes while still covering all necessary details.
   - When appropriate, consolidate sequential yes/no questions into a single, multi-option question.

3. **Mixed Answer Types (QCM & QCS):**
   - For questions where both multiple-choice (TYPE_QCM) and single-choice (TYPE_QCS) answers are relevant (e.g., when one option represents “no complication”), include both answer types as siblings.
   - **Example:**
     Do you have one or more anomalies of one of the rotator cuff tendons?
         - supraspinatus (TYPE_QCM)
         - infraspinatus (TYPE_QCM)
         - subscapularis (TYPE_QCM)
         - long biceps tendon (TYPE_QCM)
         - rotator cuff interval (TYPE_QCM)
         - small circle (TYPE_QCM)
         - no tendon abnormality (TYPE_QCS)

4. **Efficient Node Structuring:**
   - Aim to reduce redundant nodes by combining related clinical details.
   - *Strictly* Do not exceed 5 answer options/child per question/node. If more are needed, list the five most common answers and include an "Other" option/child as the 6th child as sub-question listing less frequent answers.
     (e.g., "Do you have pain?", "Do you have swelling?", "Do you have stiffness?", consider asking a single question like "What symptoms have led you to consult?" with answers such as pain, stiffness, and swelling as answers to this question) now this doesnt mean you should not as yes or no questions.
    for eg :
      What osteoarticular signs of glenohumeral instability do you notice?

        - Bankart bone lesion
        - GLAD lesion (Anterior inferior labral fissure + anteroinferior glenoid cartilage lesion)
        - posterosuper… Hill-Sachs (or Malgaigne) notch
        - bone avulsion of the humeral insertion of the LGHI (BHAGL)

        here instead asking each question separately with a yes or no answer make it into a single question with all the answers as options. but u can ask yes or no questions if applicable.

5. **Mandatory Section:**
   - Every tree must include a "Symptoms Motivating Examination:" section with relevant clinical options.

**Return Format:**

      1. SECTIONS
        create each major section as:
        - "<Section Title>" (TYPE_TITLE)

      2. QUESTIONS & OPTIONS
        Under each TYPE_TITLE, build a sequence of questions and answer options:
        a. Ask a clinical question:
            - "<Question text>?" (TYPE_QUESTION)
        b. List its answer choices:
            - "<Option A>" (TYPE_QCS) or (TYPE_QCM)
            - "<Option B>" (TYPE_QCS) or (TYPE_QCM)
            …
        c. If any option requires further detail, branch again with:
            - "<Follow-up question>?" (TYPE_QUESTION)
              * Under that, list its choices or fields:
                - "<Field Name>" (TYPE_QCS/QCM)
                - "<Measure Name>" (TYPE_MEASURE) (unit)
                - "<Text Field>" (TYPE_TEXT)


      4. PARENT QUESTION FOR MEASURE/TEXT
         **Strictly Whenever you include a TYPE_MEASURE or TYPE_TEXT leaf, first insert its parent question (TYPE_QUESTION) with the same label for each one of it**.
          Node Text: (Nodetype)
          If the nodetype is TYPE_MEASURE, include the unit in parentheses (e.g., `(years)`, `(mm)`).
           eg : Duration: (TYPE_MEASURE) (days)
      5. Phrase in the leaf nodes :
        For every leaf node (i.e. a node of type TYPE_QCM, TYPE_QCS, TYPE_MEASURE, TYPE_DATE or TYPE_TEXT with no children), append a pipe and a quoted, radiology-style summary phrase immediately after its label,
        for example format : Duration: (TYPE_MEASURE) (days)  |  "Antécédent de la durée de la condition en jours."
                            Mild (TYPE_QCS) |  "Antécédent de fièvre légère."

**Allowed Node Types:**
- TYPE_TITLE
- TYPE_QUESTION
- TYPE_QCM (multiple choice)
- TYPE_QCS (single choice)
- TYPE_MEASURE
- TYPE_DATE
- TYPE_TEXT (free text)

**Example Output Structure (One-Shot):**
INDICATION (TYPE_TITLE)
    Quel est le motif principal de l’examen ? (TYPE_TITLE)
        Douleur (TYPE_QCM)  |  "Douleur signalée."
            Où se situe la douleur ? (TYPE_QUESTION)
                Région anatomique (TYPE_TEXT)  |  "Localisation de la douleur spécifiée."
            Depuis combien de temps la douleur persiste-t-elle ? (TYPE_QUESTION)
                Durée de la douleur (TYPE_MEASURE) (jours)  |  "de douleur"
        Traumatisme (TYPE_QCM)  |  "Traumatisme suspecté."
            Type de traumatisme suspecté ? (TYPE_QUESTION)
                Contusion (TYPE_QCS)          |  "Contusion suspectée."
                Fracture (TYPE_QCS)           |  "Fracture suspectée."
                Autre traumatisme (TYPE_QCS)  |  "Autre traumatisme signalé."
                    Précisez le traumatisme ? (TYPE_QUESTION)
                        Détails du traumatisme (TYPE_TEXT)  |  "Description du traumatisme fournie."
        Infection (TYPE_QCM)  |  "Infection suspectée."
            Quels symptômes infectieux ? (TYPE_QUESTION)
                Fièvre (TYPE_QCS)                    |  "Fièvre signalée."
                Écoulement purulent (TYPE_QCS)       |  "Écoulement purulent signalé."
                Autre symptôme infectieux (TYPE_QCS) |  "Autre symptôme infectieux signalé."
                    Précisez le symptôme ? (TYPE_QUESTION)
                        Détails du symptôme (TYPE_TEXT)  |  "Description du symptôme infectieux fournie."
        Bilan de suivi (TYPE_QCM)  |  "Examen de suivi demandé."
            Intervalle depuis le dernier examen ? (TYPE_QUESTION)
                Intervalle de suivi (TYPE_MEASURE) (mois)  |  " depuis le dernier examen"
        Autre indication (TYPE_QCM)  |  "Autre indication signalée."
            Précisez l’indication ? (TYPE_QUESTION)
                Détails de l’indication (TYPE_TEXT)  |  "Détails de l’indication fournis."




**Warnings:**
- Produce only one top-level "INDICATION" node.
- Do not include extraneous text or duplicate node names.
- **Logical Consistency:** Ensure that every question and option is created with careful clinical reasoning to avoid redundant or unnecessary nodes.

**Context Dump:**
- This INDICATION tree is for a radiological exam related to "{self.file_type}" and the diseases: {', '.join(self.disease_context)}.
- The structure should support multiple answer types (MCQ, SCQ, numerical, date, free text) and incorporate logical or calculation nodes where needed.
- Emphasis is on the quality and efficiency of the clinical questions and answers—each node should be deliberate, reducing overall node count without sacrificing clinical depth.
"""
            if expanded_prompt is not None and evaluation_feedback:
                system_instruction += f"\n\n**Evaluation Feedback from Previous Iteration:**\n{evaluation_feedback}\nPlease incorporate these improvements."
            if iteration == 0:
                user_prompt = f"""
**Goal:**
Generate an initial high-level INDICATION section for a radiological exam strictly related to "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}.
Establish major categories (e.g., Patient Information and a mandatory "Symptoms Motivating Examination:" section) with minimal detail.

**Return Format:**
- A single top-level "INDICATION" node (zero indentation).
- All subordinate nodes indented by 4 spaces and formatted as: Node Text: (Nodetype)

**Warnings:**
- Produce only one top-level "INDICATION" node.
- Include the "Symptoms Motivating Examination:" section with relevant clinical options.
- *Strictly* Avoid duplicating node names at the same hierarchical level and no need to add eg text in node text and do not include any extraneous text.

**Context Dump:**
- This outline is for a radiological exam pertaining to "{self.file_type}" and diseases: {', '.join(self.disease_context)}.
- Emphasize quality and efficiency: focus on logically sound clinical questions that minimize unnecessary nodes.
"""
            elif iteration < self.indication_iterations - 1:
                user_prompt = f"""
**Goal:**
Refine and expand the existing INDICATION section {expanded_prompt} by adding more depth and detail. Enhance disease-specific and symptom-related branches with additional high-quality clinical questions and answer options.

**Return Format:**
- Maintain one top-level "INDICATION" node with sub-nodes indented by 4 spaces.
- Format each node as: Node Text: (Nodetype)

**Warnings:**
- Avoid duplicating node names at the same level.
"""
            else:
                user_prompt = f"""
**Goal:**
Fully complete and polish the INDICATION section {expanded_prompt} by incorporating deep sub-nodes for all underdeveloped branches **Focus on addressing the evaluation feedback**. Ensure every clinically relevant query is addressed with efficient, non-redundant nodes.

**Return Format:**
- Retain one top-level "INDICATION" node with sub-nodes indented by 4 spaces.
- Each node must be formatted as: Node Text (Nodetype)

**Warnings:**
- Do not add any new top-level nodes.
"""
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]
            response = self.model.invoke(messages)
            expanded_prompt = self.extract_section(response.content)
            evaluation_feedback = self.evaluate_tree_indication(expanded_prompt)
            print(f"Evaluation feedback (iteration {iteration+1}):")
            print(evaluation_feedback)
        print(f"Final INDICATION tree text length: {len(expanded_prompt)}")

        return expanded_prompt

    def generate_technical_tree(self, indication_tree_text: str) -> str:
        technical_tree = None
        evaluation_feedback = ""
        for iteration in range(self.technical_iterations):
            print(f"Generating TECHNICAL tree: iteration {iteration+1}...")
            system_instruction = f"""
**Goal:**
You are a medical professional. Your goal is to generate a structured, hierarchical TECHNICAL tree for a radiological exam. This tree should detail the technical parameters and protocols used during imaging—such as contrast injection usage, imaging sequences (e.g., T1, T2, FLAIR, angiographic sequences), and other modality-specific settings. It must be strictly tailored to the file type "{self.file_type}" and the following diseases: {', '.join(self.disease_context)}.
The TECHNICAL tree is provided in the context of the INDICATION tree below:
{indication_tree_text}
but should not duplicate information from the INDICATION or RESULT trees.

**User Input (Must Follow Strictly):**
{self.user_input}

**Strict Guidelines and Additional Instructions:**
1. **Non-Empty, Contextual Nodes:**
   - Every node must include clear, context-specific text. Avoid empty or vague nodes.
2. **High-Quality Clinical Options:**
   - generated content should be in french expcept for the node type and unit which are in the brackets.
   - Craft questions and answer options with logical clinical reasoning. Ensure each question is efficient, consolidating related inquiries to reduce redundant nodes.
   - **Mixed Answer Types (QCM & QCS):**
   - For questions where both multiple-choice (TYPE_QCM) and single-choice (TYPE_QCS) answers are relevant (e.g., when one option represents “no complication”), include both answer types as siblings.
   - **Example:**
     Do you have one or more anomalies of one of the rotator cuff tendons?
         - supraspinatus (TYPE_QCM)
         - infraspinatus (TYPE_QCM)
         - subscapularis (TYPE_QCM)
         - long biceps tendon (TYPE_QCM)
         - rotator cuff interval (TYPE_QCM)
         - small circle (TYPE_QCM)
         - no tendon abnormality (TYPE_QCS)

3. **Efficient Node Structuring:**
   - *Strictly* Do not exceed 5 answer options/child per question/node. If more are needed, list the five most common answers and include an "Other" option/child as the 6th child as sub-question listing less frequent answers.
     (e.g., "Do you have pain?", "Do you have swelling?", "Do you have stiffness?", consider asking a single question like "What symptoms have led you to consult?" with answers such as pain, stiffness, and swelling as answers to this question) now this doesnt mean you should not as yes or no questions.
    for eg :
      What osteoarticular signs of glenohumeral instability do you notice?

        - Bankart bone lesion
        - GLAD lesion (Anterior inferior labral fissure + anteroinferior glenoid cartilage lesion)
        - posterosuper… Hill-Sachs (or Malgaigne) notch
        - bone avulsion of the humeral insertion of the LGHI (BHAGL)

        here instead asking each question separately with a yes or no answer make it into a single question with all the answers as options. but u can ask yes or no questions if applicable.

4. **Logical Grouping:**
   - Group related technical parameters logically (e.g., injection protocols, imaging sequences, additional parameters like coil type or slice thickness).


 **Return Format:**

      1. SECTIONS
        create each major section as:
        - "<Section Title>" (TYPE_TITLE)

      2. QUESTIONS & OPTIONS
        Under each TYPE_TITLE, build a sequence of questions and answer options:
        a. Ask a clinical question:
            - "<Question text>?" (TYPE_QUESTION)
        b. List its answer choices:
            - "<Option A>" (TYPE_QCS) or (TYPE_QCM)
            - "<Option B>" (TYPE_QCS) or (TYPE_QCM)
            …
        c. If any option requires further detail, branch again with:
            - "<Follow-up question>?" (TYPE_QUESTION)
              * Under that, list its choices or fields:
                - "<Field Name>" (TYPE_QCS/QCM)
                - "<Measure Name>" (TYPE_MEASURE) (unit)
                - "<Text Field>" (TYPE_TEXT)


      4. PARENT QUESTION FOR MEASURE/TEXT
         **Strictly Whenever you include a TYPE_MEASURE or TYPE_TEXT leaf, first insert its parent question (TYPE_QUESTION) with the same label for each one of it**.
      - Every node must be formatted as:
          Node Text (Nodetype)
          If the nodetype is TYPE_MEASURE, include the unit in parentheses (e.g., `(years)`, `(mm)`).
           eg : Duration: (TYPE_MEASURE) (days)
      5. Phrase in the leaf nodes :
        For every leaf node (i.e. a node of type TYPE_QCM, TYPE_QCS, TYPE_MEASURE, TYPE_DATE or TYPE_TEXT with no children), append a pipe and a quoted, radiology-style summary phrase immediately after its label,
        for example format : Duration (TYPE_MEASURE) (days)  |  "Antécédent de la durée de la condition en jours."
                            Mild (TYPE_QCS) |  "Antécédent de fièvre légère."

**Allowed Node Types:**
- TYPE_TITLE
- TYPE_QUESTION
- TYPE_QCM (multiple choice)
- TYPE_QCS (single choice)
- TYPE_MEASURE
- TYPE_DATE
- TYPE_TEXT (free text)

**Example Output Structure (One-Shot):**
       TECHNIQUE (TYPE_TITLE)
          Détails de la technique (TYPE_TITLE)
              Quelle modalité d’imagerie a été utilisée ? (TYPE_QUESTION)
                  Échographie (TYPE_QCS)  |  "Échographie réalisée."
                      Quel transducteur a été utilisé ? (TYPE_QUESTION)
                          Transducteur (TYPE_TEXT)  |  "Transducteur spécifié."
                      Quelle fréquence de sonde ? (TYPE_QUESTION)
                          Fréquence (TYPE_MEASURE) (MHz)  |  " MHz de fréquence de sonde."
                      Doppler appliqué ? (TYPE_QUESTION)
                          Oui (TYPE_QCS)  |  "Doppler appliqué."
                          Non (TYPE_QCS)  |  "Doppler non appliqué."
                  Scanner (TYPE_QCS)  |  "Scanner réalisé."
                      Quel voltage a été utilisé ? (TYPE_QUESTION)
                          Voltage (TYPE_MEASURE) (kV)  |  " kV de tension."
                      Quel débit a été utilisé ? (TYPE_QUESTION)
                          Débit (TYPE_MEASURE) (mAs)  |  " mAs de débit."
                      Contraste iodé injecté ? (TYPE_QUESTION)
                          Oui (TYPE_QCS)  |  "Contraste iodé injecté."
                              Quel volume de contraste ? (TYPE_QUESTION)
                                  Volume (TYPE_MEASURE) (mL)  |  " mL de contraste injecté."
                          Non (TYPE_QCS)  |  "Pas de contraste iodé injecté."
                  IRM (TYPE_QCS)  |  "IRM réalisée."
                      Séquences utilisées ? (TYPE_QUESTION)
                          Séquences (TYPE_TEXT)  |  "Séquences spécifiées."
                  Autre modalité (TYPE_QCS)  |  "Autre modalité précisée."
                      Précisez la modalité ? (TYPE_QUESTION)
                          Modalité (TYPE_TEXT)  |  "Modalité précisée."


**Warnings:**
- Produce only one top-level "TECHNICAL" node with zero indentation.
- *Strictly* Avoid duplicating node names at the same hierarchical level and no need to add eg text in node text and do not include any extraneous text.
- Do not include any extraneous output (such as quotes or additional text) beyond the structured tree.

**Context Dump:**
- The TECHNICAL tree is intended to capture all relevant imaging protocols for a radiological exam of type "{self.file_type}" in the context of diseases: {', '.join(self.disease_context)}.
- This structure should clearly document technical details like contrast usage, specific imaging sequences, and additional parameters, ensuring consistency with the INDICATION and RESULT trees.
"""
            if technical_tree is not None and evaluation_feedback:
                system_instruction += f"\n\n**Evaluation Feedback from Previous Iteration:**\n{evaluation_feedback}\nPlease incorporate these improvements."
            user_prompt = f"""
**Goal:**
Generate a single, structured TECHNICAL section for a radiological exam based on imaging protocols. Tailor the output to the file type "{self.file_type}" and diseases: {', '.join(self.disease_context)}.

**Return Format:**
- One top-level "TECHNICAL" node (zero indentation) with sub-nodes indented by 4 spaces.
- Each node must include its label followed by its nodetype (e.g., "Injection Protocol: (TYPE_TOPIC)").

**Warnings:**
- Produce only one top-level "TECHNICAL" node.
- Do not duplicate node names at the same hierarchical level.
- Do not include any extraneous text outside the structured tree.

**Context Dump:**
- The TECHNICAL tree should capture detailed imaging protocols including contrast usage, specific imaging sequences, and additional technical parameters. It should be logically grouped, comprehensive, and efficient.
"""
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]
            response = self.model.invoke(messages)
            technical_tree = self.extract_section(response.content)
            evaluation_feedback = self.evaluate_tree_technical(technical_tree)
            print(f"Evaluation feedback (iteration {iteration+1}):")
            print(evaluation_feedback)
        print(f"Final TECHNICAL tree text length: {len(technical_tree)}")
        return technical_tree

    def generate_result_tree(self, indication_tree_text: str, technical_tree_text: str) -> str:
        result = None
        evaluation_feedback = ""
        for iteration in range(self.result_iterations):
            print(f"Generating RESULT tree: iteration {iteration+1}...")
            if result is not None and evaluation_feedback:
                system_instruction += f"\n\n**Evaluation Feedback from Previous Iteration:**\n{evaluation_feedback}\nPlease incorporate these improvements."
            if iteration == 0:
                user_prompt = f"""
    **Goal:**
    Generate an initial high-level RÉSULTAT section for a radiological exam that captures final imaging observations. Tailor the output to "{self.file_type}" and diseases: {', '.join(self.disease_context)}.
    Establish major anatomical categories (e.g., pleura, parenchyma, mediastinum, bones, devices) with minimal detail.
    Include an outline that differentiates the findings.

      For context, reference the INDICATION tree below and the TECHNICAL tree below:
    INDICATION:
    {indication_tree_text}
    TECHNICAL:
    {technical_tree_text}
    —but do not duplicate their content.
    """
            elif iteration < self.result_iterations - 1:
                user_prompt = f"""
    **Goal:**
    Refine and expand the existing RÉSULTAT section {result} by adding more detailed sub-nodes for each anatomical category. Emphasize further elaboration of abnormal findings, including measurements and descriptive details.
    Ensure the structure is efficient and avoid duplicating node names at the same hierarchical level.

    """
            else:
                user_prompt = f"""
    **Goal:**
    Fully complete and polish the RÉSULTAT section {result} by incorporating deep sub-nodes for each anatomical category. Detail abnormalities by specifying size, extent, severity, and anatomical locations. Include measurement, logical, or calculation nodes as needed.
    Ensure the final output is comprehensive and structured as a single RESULT tree.
    """
            system_instruction = f"""
    **Goal:**
    You are a medical professional. Your task is to generate a structured, hierarchical RESULT tree for a radiological exam. This tree must document final imaging observations by detailing anatomical structures (e.g., pleura, parenchyma, mediastinum, bones, devices) and any detected abnormalities (e.g., effusions, nodules, calcifications). It must be strictly tailored to the file type "{self.file_type}" and diseases: {', '.join(self.disease_context)}.


    **User Input (Must Follow Strictly):**
    {self.user_input}

    **Strict Guidelines and Additional Instructions:**
1. **Non-Empty, Contextual Nodes:**
   - Every node must include clear, context-specific text. Avoid empty or vague nodes.
2. **High-Quality Clinical Options:**
   - generated content should be in french expcept for the node type and unit which are in the brackets.
   - Craft questions and answer options with logical clinical reasoning. Ensure each question is efficient, consolidating related inquiries to reduce redundant nodes.
   - **Mixed Answer Types (QCM & QCS):**
   - For questions where both multiple-choice (TYPE_QCM) and single-choice (TYPE_QCS) answers are relevant (e.g., when one option represents “no complication”), include both answer types as siblings.
   - **Example:**
     Do you have one or more anomalies of one of the rotator cuff tendons?
         - supraspinatus (TYPE_QCM)
         - infraspinatus (TYPE_QCM)
         - subscapularis (TYPE_QCM)
         - long biceps tendon (TYPE_QCM)
         - rotator cuff interval (TYPE_QCM)
         - small circle (TYPE_QCM)
         - no tendon abnormality (TYPE_QCS)

3. **Efficient Node Structuring:**
   - *Strictly* Do not exceed 5 answer options/child per question/node. If more are needed, list the five most common answers and include an "Other" option/child as the 6th child as sub-question listing less frequent answers.
     (e.g., "Do you have pain?", "Do you have swelling?", "Do you have stiffness?", consider asking a single question like "What symptoms have led you to consult?" with answers such as pain, stiffness, and swelling as answers to this question) now this doesnt mean you should not as yes or no questions.
    for eg :
      What osteoarticular signs of glenohumeral instability do you notice?

        - Bankart bone lesion
        - GLAD lesion (Anterior inferior labral fissure + anteroinferior glenoid cartilage lesion)
        - posterosuper… Hill-Sachs (or Malgaigne) notch
        - bone avulsion of the humeral insertion of the LGHI (BHAGL)

        here instead asking each question separately with a yes or no answer make it into a single question with all the answers as options. but u can ask yes or no questions if applicable.


    **Return Format:**
    - Produce exactly one top-level "RÉSULTAT" node (zero indentation) with all sub-nodes indented by 4 spaces.
    -  **Return Format:**

      1. SECTIONS
        create each major section as:
        - "<Section Title>" (TYPE_TITLE)

      2. QUESTIONS & OPTIONS
        Under each TYPE_TITLE, build a sequence of questions and answer options:
        a. Ask a clinical question:
            - "<Question text>?" (TYPE_QUESTION)
        b. List its answer choices:
            - "<Option A>" (TYPE_QCS) or (TYPE_QCM)
            - "<Option B>" (TYPE_QCS) or (TYPE_QCM)
            …
        c. If any option requires further detail, branch again with:
            - "<Follow-up question>?" (TYPE_QUESTION)
              * Under that, list its choices or fields:
                - "<Field Name>" (TYPE_QCS/QCM)
                - "<Measure Name>" (TYPE_MEASURE) (unit)
                - "<Text Field>" (TYPE_TEXT)


      4. PARENT QUESTION FOR MEASURE/TEXT
        **Strictly Whenever you include a TYPE_MEASURE or TYPE_TEXT leaf, first insert its parent question (TYPE_QUESTION) with the same label for each one of it**.
      - Every node must be formatted as:
          Node Text (Nodetype)
          If the nodetype is TYPE_MEASURE, include the unit in parentheses (e.g., `(years)`, `(mm)`).
           eg : Duration (TYPE_MEASURE) (days)
      5. Phrase in the leaf nodes :
        For every leaf node (i.e. a node of type TYPE_QCM, TYPE_QCS, TYPE_MEASURE, TYPE_DATE or TYPE_TEXT with no children), append a pipe and a quoted, radiology-style summary phrase immediately after its label,
        for example format : Duration: (TYPE_MEASURE) (days)  |  "Antécédent de la durée de la condition en jours."
                            Mild (TYPE_QCS) |  "Antécédent de fièvre légère."

    **Example Output Structure (One-Shot):**
          RÉSULTAT (TYPE_TITLE)
              Morphologie et dimensions (TYPE_TITLE)
                  Quelle est la taille du lobe droit ? (TYPE_QUESTION)
                      Taille du lobe droit (TYPE_MEASURE) (mm)  |  " de lobe droit"
                  Quelle est la taille du lobe gauche ? (TYPE_QUESTION)
                      Taille du lobe gauche (TYPE_MEASURE) (mm)  |  "de lobe gauche"
                  Quelle est l’épaisseur de l’isthme ? (TYPE_QUESTION)
                      Épaisseur de l’isthme (TYPE_MEASURE) (mm)  |  " d’isthme"
                  Quel est le volume global de la thyroïde ? (TYPE_QUESTION)
                      Volume global de la thyroïde (TYPE_MEASURE) (mL)  |  "de volume thyroïdien"
              Présence d’anomalies morphologiques (TYPE_TITLE)
                  Oui (TYPE_QCS)  |  "Anomalie morphologique présente"
                  Non (TYPE_QCS)  |  "Pas d’anomalie morphologique"
                  Veuillez décrire l’anomalie morphologique (TYPE_QUESTION)
                      Description de l'anomalie (TYPE_TEXT)  |  "Anomalie morphologique décrite"
                  Quel type d’anomalie morphologique ? (TYPE_QUESTION)
                      Agénésie (TYPE_QCS)    |  "Agénésie"
                      Hémiagénésie (TYPE_QCS)   |  "Hémiagénésie"
                      Lobe pyramidal proéminent (TYPE_QCS)  |  "Lobe pyramidal proéminent"
                      Thyroïde ectopique (TYPE_QCS)  |  "Thyroïde ectopique"
                      Autre (TYPE_QCS)   |  "Autre anomalie morphologique"
                      Précisez l’anomalie morphologique (TYPE_QUESTION)
                          Précisez l’anomalie (TYPE_TEXT)    |  "Détail de l’anomalie fourni"
              Échostructure du parenchyme thyroïdien (TYPE_TITLE)
                  Homogène (TYPE_QCS)  |  "Parenchyme homogène"
                  Hétérogène (TYPE_QCS)   |  "Parenchyme hétérogène"
                  Veuillez décrire l’hétérogénéité (TYPE_QUESTION)
                      Description de l'hétérogénéité (TYPE_TEXT)  |  "Hétérogénéité décrite"
                  Quel type d’hétérogénéité ? (TYPE_QUESTION)
                      Focalisée (TYPE_QCS)   |  "Hétérogénéité focalisée"
                      Diffuse (TYPE_QCS)   |  "Hétérogénéité diffuse"
                      Mixte (TYPE_QCS)    |  "Hétérogénéité mixte"




    **Warnings:**
    - Produce only one top-level "RÉSULTAT" node; it must have zero indentation.
    - *Strictly* Avoid duplicating node names at the same hierarchical level and no need to add eg text in node text and do not include any extraneous text.
    - Do not output any extraneous text or commentary beyond the structured tree.
    - **Strictly do not output anything other than the structured tree.**

    **Context Dump:**
    - The RÉSULTAT tree should capture final imaging observations, potentially referencing details from the INDICATION and TECHNICAL trees.
    - It must document both normal and abnormal findings, including measurement values, severity, and specific anatomical locations.
    - This structure supports a comprehensive radiological report that is both systematic and clinically informative.
    """
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]
            response = self.model.invoke(messages)
            result = self.extract_section(response.content)
            evaluation_feedback = self.evaluate_tree_result(result)
            print(f"Evaluation feedback (iteration {iteration+1}):")
            print(evaluation_feedback)
        print(f"Final RESULT tree text length: {len(result)}")
        return result

    # ---------------- Parsing, Deduplication, Transformation, and Combining ----------------

    def translate_to_french(self, text) :
      translator = GoogleTranslator(source = "en",target = 'fr')
      try :
        translated_test = translator.translate(text)
        return translated_test
      except Exception as e :
        return f"Transaltion failed : {e}"

    def parse_indentation_tree(self, tree_str: str) -> list:
        """
        Convert an indented tree string into a list of node dicts,
        extracting a 'phrase' for leaf nodes after the pipe (|).
        """
        import re

        lines = tree_str.splitlines()
        stack = []           # holds tuples of (node_dict, indent_level)
        nodes_list = []

        recognized_node_types = {
             "TYPE_TITLE", "TYPE_QUESTION",
            "TYPE_QCM", "TYPE_QCS", "TYPE_MEASURE",
            "TYPE_DATE", "TYPE_TEXT"
        }

        for line in lines:
            if not line.strip():
                continue
            # determine indentation
            indent = len(line) - len(line.lstrip(" "))
            content = line.strip()

            # split off report phrase if present
            if "|" in content:
                label_part, phrase_part = [p.strip() for p in content.split("|", 1)]
                phrase = phrase_part.strip('"')
            else:
                label_part = content
                phrase = None

            # extract bracketed node type and optional unit
            bracket_texts = re.findall(r'\([^()]*\)', label_part)
            node_type_extracted = None
            unit_extracted = None
            new_text = label_part
            for bt in bracket_texts:
                inside = bt[1:-1].strip()
                if inside in recognized_node_types:
                    node_type_extracted = inside
                else:
                    if unit_extracted is None:
                        unit_extracted = inside
                new_text = new_text.replace(bt, "", 1)
            new_text = new_text.rstrip(":").strip()

            # decide nodeType
            if node_type_extracted:
                node_type = node_type_extracted
            else:
                # fallback: root if top-level, else question or generic
                if not stack:
                    node_type = "root"
                # else:
                #     node_type = "TYPE_QUESTION" if new_text.endswith("?") else "node"

            # pop stack until finding proper parent indent
            while stack and indent <= stack[-1][1]:
                stack.pop()

            parent_id = stack[-1][0]["id"] if stack else None
            parent_text = stack[-1][0]["text"] if stack else None

            # assign unique ID
            node_id = str(self.node_counter)
            self.node_counter += 1

            # build node dict, including the new 'phrase' field
            node = {
                "id": node_id,
                "nodeType": node_type,
                "text": new_text,
                "isLeaf": True,
                "parent": parent_id,
                "parentText": parent_text,
                "childs": [],
                "unit": unit_extracted if node_type == "TYPE_MEASURE" else None,
                "phrase": phrase
            }
            nodes_list.append(node)

            # attach to parent if exists
            if parent_id:
                stack[-1][0]["childs"].append(node_id)
                stack[-1][0]["isLeaf"] = False

            # push current node onto stack
            stack.append((node, indent))

            # --- build lookup for parent node types ---
        id_to_node = {n["id"]: n for n in nodes_list}

        # --- inject synthetic TYPE_QUESTION parents for TEXT/MEASURE/DATE leaves without a QUESTION parent ---
        extra_nodes = []
        for leaf in list(nodes_list):
            parent_id = leaf.get("parent")
            parent_type = id_to_node.get(parent_id, {}).get("nodeType")
            if leaf["nodeType"] in ("TYPE_TEXT", "TYPE_MEASURE", "TYPE_DATE") and parent_type != "TYPE_QUESTION":
                # create a new question node
                q_id = str(self.node_counter)
                self.node_counter += 1
                question_node = {
                    "id": q_id,
                    "nodeType": "TYPE_QUESTION",
                    "text": leaf["text"],
                    "isLeaf": False,
                    "parent": parent_id,
                    "parentText": leaf.get("parentText"),
                    "childs": [leaf["id"]],
                    "unit": None,
                    "phrase": None
                }
                # rewire the original parent to point to this question node
                if parent_id:
                    parent_node = id_to_node[parent_id]
                    parent_node["childs"] = [
                        q_id if c == leaf["id"] else c
                        for c in parent_node["childs"]
                    ]
                    parent_node["isLeaf"] = False

                # re-parent the leaf under the new question node
                leaf["parent"] = q_id
                leaf["parentText"] = question_node["text"]

                extra_nodes.append(question_node)

        # append all synthetic question nodes
        nodes_list.extend(extra_nodes)

        # rebuild lookup now that we have extra nodes
        id_to_node = {n["id"]: n for n in nodes_list}

        # --- re-type any QCS/QCM nodes that have QCS/QCM children into TYPE_QUESTION ---
        for node in nodes_list:
            if node["nodeType"] in ("TYPE_QCS", "TYPE_QCM"):
                for child_id in node["childs"]:
                    child = id_to_node.get(child_id)
                    if child and child["nodeType"] in ("TYPE_QCS", "TYPE_QCM"):
                        node["nodeType"] = "TYPE_QUESTION"
                        node["isLeaf"] = False
                        break

        return nodes_list



    def deduplicate_nodes(self, nodes_list: list) -> tuple:
        """
        Merge nodes that have the same structure based on a signature.
        Returns a tuple (dedup_node_dict, alias_mapping).
        """
        node_dict = {node["id"]: node for node in nodes_list}
        memo = {}
        signature_map = {}
        alias_mapping = {}
        def get_signature(node_id):
            if node_id in memo:
                return memo[node_id]
            node = node_dict[node_id]
            child_signatures = tuple(get_signature(child_id) for child_id in node["childs"])
            signature = (node["text"], node["nodeType"], node["parentText"], node.get("unit"), child_signatures)
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
        """
        Transform deduplicated nodes so that the "parent" key becomes an object with "id" and "text",
        and each child in "childs" becomes an object with "id", "text", and preserves the "phrase" field.
        """
        transformed = {}
        for node_id, node in nodes_dict.items():
            new_node = {
                "id": node["id"],
                "nodeType": node["nodeType"],
                "text": node["text"],
                "isLeaf": node["isLeaf"],
                "parent": None,
                "childs": [],
                "unit": node.get("unit", None),
                "phrase": node.get("phrase", None)   # preserve the phrase field
            }
            # set parent object if exists
            if node.get("parent") and node["parent"] in nodes_dict:
                new_node["parent"] = {
                    "id": node["parent"],
                    "text": nodes_dict[node["parent"]]["text"]
                }
            # build child objects
            for child_id in node["childs"]:
                if child_id in nodes_dict:
                    child_node = nodes_dict[child_id]
                    child_obj = {
                        "id": child_id,
                        "text": child_node["text"],
                        "phrase": child_node.get("phrase", None)  # carry child's phrase too
                    }
                    new_node["childs"].append(child_obj)
            transformed[node_id] = new_node
        return transformed

    def combine_trees(self, indication_nodes: list, technical_nodes: list, result_nodes: list) -> dict:
        def get_root(nodes):
            for node in nodes:
                if node.get("parent") is None and node.get("nodeType") == "TYPE_TITLE":

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
            "text": self.translate_to_french(self.file_type),
            "isLeaf": False,
            "parent": None,
            "parentText": None,
            "childs": []
        }
        root_label_id = str(self.node_counter)
        self.node_counter += 1
        new_root_label = {
            "id": root_label_id,
            "nodeType": "TYPE_TITLE",
            "text": self.translate_to_french(self.file_type),
            "isLeaf": True,
            "parent": new_root_id,
            "parentText": self.translate_to_french(self.file_type),
            "childs": []
        }
        new_root["childs"].append(new_root_label["id"])
        nolabel = []
        for i in range(4) :
          new_id = str(self.node_counter)
          new_blank_node  = {
            "id": new_id,
            "nodeType": "TYPE_TITLE",
            "text": "BLANK",
            "isLeaf": True,
            "parent": new_root_id,
            "parentText": self.translate_to_french(self.file_type),
            "childs": []
          }
          nolabel.append(new_blank_node)
          new_root["childs"].append(new_blank_node["id"])
          self.node_counter += 1
          if indication_root and i == 0:
            indication_root["text"] = "INDICATION"
            indication_root["parent"] = new_root_id
            indication_root["parentText"] = self.file_type
            new_root["childs"].append(indication_root["id"])
          if technical_root and i == 1:
              technical_root["text"] = "TECHNIQUE"
              technical_root["parent"] = new_root_id
              technical_root["parentText"] = self.file_type
              new_root["childs"].append(technical_root["id"])
          if result_root and i == 2:
              result_root["text"] = "RÉSULTAT"
              result_root["parent"] = new_root_id
              result_root["parentText"] = self.file_type
              new_root["childs"].append(result_root["id"])

        combined_nodes = [new_root] + [new_root_label] + [nolabel[0]] + indication_nodes + [nolabel[1]] + technical_nodes + [nolabel[2]] + result_nodes + [nolabel[3]]
        return { node["id"]: node for node in combined_nodes }

    # ---------------- Run Pipeline ----------------

    def run(self):
        print("Generating INDICATION tree...")
        indication_text = self.generate_indication_tree()
        print("Generated INDICATION Tree Text:")
        print(indication_text)

        print("Generating TECHNICAL tree...")
        technical_text = self.generate_technical_tree(indication_text)
        print("Generated TECHNICAL Tree Text:")
        print(technical_text)

        print("Generating RESULT tree...")
        result_text = self.generate_result_tree(indication_text, technical_text)
        print("Generated RESULT Tree Text:")
        print(result_text)

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

        print(f"Length of combined tree: {len(transformed_nodes)}")
        print("Pipeline completed successfully.")

        return list(transformed_nodes.values())
                    
        # with open(self.combined_json_filename, "w") as f:
        #     json.dump(list(transformed_nodes.values()), f, indent=2)
        # print(f"Combined JSON saved to {self.combined_json_filename}")

        # # self.plot_tree(transformed_nodes, self.combined_png_filename)
        # print(f"Length of combined tree: {len(transformed_nodes)}")
        # stream_lit_text.text("Completed and processed generated tree")