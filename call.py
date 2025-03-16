import streamlit as st
import pymongo
from bson import ObjectId
from treeGenerator import CombinedMedicalTreeGenerator
from custom2doctreen_parser import CustomToDoctreenConverter

URI = st.secrets["general"]["uri"]

def main():
    doctreen_icon = "https://static.wixstatic.com/media/cb6226_4224827f5f13449ebb1ce7b71abbbc10%7Emv2.png/v1/fill/w_192%2Ch_192%2Clg_1%2Cusm_0.66_1.00_0.01/cb6226_4224827f5f13449ebb1ce7b71abbbc10%7Emv2.png"
    doctreen_logo = "https://static.wixstatic.com/media/cb6226_9226c5ad3a1a48e9abb5adbf8e8eb30a~mv2.png/v1/crop/x_53,y_0,w_1223,h_439/fill/w_291,h_104,fp_0.50_0.50,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/Logo%20horizontal%20fond%20blanc.png"

    # Set page config
    st.set_page_config(page_title="Medical Tree Generator", page_icon=doctreen_icon)

    # Connect to MongoDB and fetch existing trees
    client = pymongo.MongoClient(URI)
    db = client["doctreen"]
    collection = db["trees"]
    query = {"ownerId": ObjectId("679fc806c5dab815f7995fb8")}
    projection = {"treeName": 1, "lineTreeId": 1, "_id": 0}
    results = collection.find(query, projection)

    # Build a list of existing trees
    generated_trees = []
    for record in results:
        tree_name = record['treeName']
        line_id = str(record['lineTreeId'])
        tree_link = f'https://front.interns.doctreen.io/edit/{line_id}'
        generated_trees.append({"name": tree_name, "link": tree_link})

    # ----------------- SIDEBAR SECTION -----------------
    st.sidebar.title("Previously Generated Trees")

    # Define custom CSS for the link buttons with hover effect
    st.sidebar.markdown(
        """
        <style>
        .treeButton {
        display: block;
        font-family: sans-serif;
        padding: 0.5em 1em;
        background-color: #088db6;
        color: white;
        text-decoration: none; /* Remove underline */
        border-radius: 4px;
        margin-bottom: 5px;
        transition: background-color 0.3s ease;
        }
        .treeButton:hover {
        background-color: #056b8a;
        color: white;          /* Keep text white on hover */
        text-decoration: none; /* Remove underline on hover */
        }
        </style>
        """, unsafe_allow_html=True
    )

    if generated_trees:
        for t in generated_trees:
            st.sidebar.markdown(
                f'<a class="treeButton" href="{t["link"]}" target="_blank" style="color: white; text-decoration: none;">{t["name"]}</a>',
                unsafe_allow_html=True
            )
    else:
        st.sidebar.write("No previously generated trees found.")


    # ----------------- MAIN PAGE SECTION -----------------
    # Display your logo and title
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="{doctreen_logo}" width="200">
        </div>
    """, unsafe_allow_html=True)
    st.title("Medical Tree Generator & Converter")

    # Input fields for new tree generation
    file_type = st.text_input("Enter file type (e.g., 'Thyroid ultrasound')", "Thyroid ultrasound")
    diseases_input = st.text_area("Enter diseases separated by commas", "Nodule Control, Echo Std, Thyroiditis")
    user_input = st.text_area("Enter addtional info if any", "")
    tree_name = st.text_input("Enter tree name", "")

    if st.button("Generate & Convert"):
        if not tree_name:
            st.error("Please enter a tree name.")
            return
        
        disease_context = [d.strip() for d in diseases_input.split(",") if d.strip()]

        st.info("Generating medical tree...")
        my_bar = st.progress(0, text="Starting Generation")
        my_text = st.empty()
        
        generator = CombinedMedicalTreeGenerator(file_type, disease_context,user_input)
        tree = generator.run(my_bar, my_text)  # <--- The method that returns your combined tree JSON
        st.success("Pipeline completed successfully.")
        
        st.info("Uploading into Doctreen...")
        my_bar.empty()

        owner_id = "679fc806c5dab815f7995fb8"
        try:
            converter = CustomToDoctreenConverter(owner_id, tree_name)
            doctreen_nodes, _, link = converter.convert_custom_to_doctreen(tree)
            
            st.success("Conversion complete!")
            st.write(f"Total nodes inserted: {len(doctreen_nodes)}")
            st.warning("Please log onto Doctreen to view the tree", icon="⚠️")
            # A button to open the newly generated tree
            st.link_button("Click here to go to the generated tree", link)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
