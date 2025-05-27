# import json
import uuid
from bson import ObjectId
import pymongo
from datetime import datetime
# from tqdm import tqdm
import streamlit as st

URI = st.secrets["general"]["uri"]

class CustomToDoctreenConverter:
    def __init__(self, owner_id, tree_name, uri=URI):
        self.owner_id = owner_id
        self.tree_name = tree_name
        self.client = pymongo.MongoClient(uri)
        self.db = self.client["doctreen"]
        self.treenodes_collection = self.db["treenodes"]
        self.trees_collection = self.db["trees"]

    def convert_custom_to_doctreen(self, custom_nodes):
        new_nodes = []
        tree_nodes = []
        idMap = {}
        root = ''
        check = 0
        nodenum = 1
        default_report = []
        
        for node in custom_nodes:
            print(f'Node {nodenum}/{len(custom_nodes)} -', end = ' ')
            nodenum += 1
            new_uuid = str(uuid.uuid4())
            idMap[node['id']] = new_uuid
            
            if node['nodeType'] == 'TYPE_ROOT' and check == 0:
                root = new_uuid
                check = 1
                
            elif node['nodeType'] == 'TYPE_ROOT' and check == 1:
                return 'INVALID ROOT', 0
            
        print("ID Map:", idMap)
        nodenum = 1
        
        for node in custom_nodes:
            node_id = ObjectId()
            tree_nodes.append(node_id)
            value_node = {}
            print('=' * 30)
            print(f'Node {nodenum}/{len(custom_nodes)}')
            nodenum += 1
            
            if node.get("nodeType", "") == 'TYPE_MEASURE':
                print('Variation detected - MESURE')
                nodetype = 'TYPE_MESURE'
                value_node['nameValue'] = node.get("text", "").lower()
                value_node['unitValue'] = node.get('unit', '')
                
                if value_node['unitValue'] is None:
                    value_node['unitValue'] = ''
                
            elif node.get("nodeType", "") in ['TYPE_TOPIC', 'TYPE_QUESTION']:
                print('Variation detected - NODE')
                nodetype = 'TYPE_NODE'
                
            else:
                nodetype = node.get("nodeType", "")
                
            marktype = {
                "MARK_SPACE": True,
                "MARK_LINE_FINISHER": False
            }
            styling = {}
            phrase_check = False
            node_alias = node.get("text", "")
            
            if (nodetype == 'TYPE_TITLE' and ('INDICATION' in node.get("text", "").upper() or 'TECHNIQUE' in node.get("text", "").upper() or 'RÉSULTAT' in node.get("text", "").upper() or 'BLANK' in node.get("text", "").upper())) or nodetype == 'TYPE_ROOT':
                if node.get("text", "") == 'BLANK':
                    node_alias = ''
                    
                else:
                    marktype['MARK_END_LINE_FINISHER'] = True
                    phrase_check = True
                
                marktype['MARK_LINE_FINISHER'] = True
                styling = {
                    "fontSize": None,
                    "fontFamily": None,
                    "textAlign": None,
                    "italic": None,
                    "stroke": None,
                    'bold': True,
                    'underline': True
                }
                default_report.append({'nodeId': idMap[node['id']], 'checked': True, 'value': None, 'collapsed': False, '_id': ObjectId()})
            
            elif nodetype not in ['TYPE_QCS', 'TYPE_QCM']:
                default_report.append({'nodeId': idMap[node['id']], 'checked': True, 'value': None, 'collapsed': False, '_id': ObjectId()})
                
            else:
                default_report.append({'nodeId': idMap[node['id']], 'checked': False, 'value': None, 'collapsed': False, '_id': ObjectId()})
                
            phrase = []
            
            if node['phrase'] or phrase_check:
                if phrase_check:
                    temp = {
                        'phraseId': str(uuid.uuid4()),
                        'text': node['text'],
                        'alias': node['text'],
                        '_id': ObjectId()
                    }
                
                else:
                    temp = {
                        'phraseId': str(uuid.uuid4()),
                        'text': node['phrase'],
                        'alias': node['phrase'],
                        '_id': ObjectId()
                    }
                    
                phrase.append(temp)
            
            new_node = {
                "_id": node_id,
                "nodeId": idMap[node['id']],
                "nodeType": nodetype,
                "fatherId": idMap[node['parent']['id']] if node.get("parent") else None,
                "alias": node_alias,
                "value": value_node,
                "markTypes": marktype,
                "styling": styling,
                "ownerId": ObjectId(self.owner_id),
                "childNodes": [idMap.get(child.get("id"), child.get("id")) for child in node.get("childs", [])],
                "labelId": None,
                "disabled": False,
                'phrases': phrase
            }
            print("Converted node:", new_node)
            result = self.treenodes_collection.insert_one(new_node)
            print("Inserted node with _id:", result.inserted_id)
            new_nodes.append(new_node)
            
        tree_id = ObjectId()
        report = [
            {
                'reportName': 'Default',
                'reportNodes': default_report,
                'labels': {},
                'subReportMap': {},
                'priority': [],
                'lastUpdate': datetime.utcnow(),
                '_id': ObjectId()
            }
        ]
        tree_doc = {
            "_id": tree_id,
            "treeName": self.tree_name,
            "tags": [],
            "treeNodeIds": tree_nodes,
            "description": self.tree_name,
            "public": True,
            "disabled": False,
            "labels": {},
            "latest": True,
            "defaultReport": {"nodes": []},
            "subTrees": [],
            "reports": report,
            "disabledReports": [],
            "createdAt": datetime.utcnow(),
            "software_version": 1,
            "lineTreeId": tree_id,
            "ownerId": ObjectId(self.owner_id),
            "rootNodeId": root,
            'lastUpdate': datetime.utcnow()
        }
        
        print('=' * 20)
        tree_result = self.trees_collection.insert_one(tree_doc)
        print("Inserted tree document with _id:", tree_result.inserted_id)
        tree_link = f'https://integrate.doctreen.com/edit/{tree_id}'
        
        return new_nodes, tree_doc, tree_link


