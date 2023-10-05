import streamlit as st

from streamlit_gallery.utils.db_helper import get_image_from_url
from sentence_transformers import SentenceTransformer, util

from multiprocessing import Pool
from PIL import Image
import requests
from io import BytesIO
import os

import greenplumpython as gp

db = gp.database(
    params={
        "host": st.secrets["db_hostname"],
        "dbname": st.secrets["db_name"],
        "user": st.secrets["db_username"],
        "port": st.secrets["db_port"],
        "password": st.secrets["db_password"],
    }
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

gp.config.print_sql = True

# Get repr of Grenplum operator vector cosine distance
cosine_distance = gp.operator("<=>")

vector = gp.type_("vector", modifier=512)

movies = db.create_dataframe(table_name="movies_emb", schema="movies")

# First, we load the respective CLIP model
model = SentenceTransformer("clip-ViT-B-32")


def main():
    st.subheader("Instruction")
    st.markdown(
        "You can find movies with similar poster by providing an poster of the movie, either by uploading an image file or providing an image URL."
    )

    c1, c2 = st.columns([2, 2])
    number_results = c1.number_input("Top-N-Search:", value=10, min_value=1, step=10)
    c1.subheader("Upload Your Poster")
    image_search_url = c1.text_input("Enter your Poster url:", key="image")
    uploaded_file = c1.file_uploader("Choose a file")
    search_button = c1.button("Search")

    if search_button:
        if uploaded_file is not None:
            img_search = Image.open(uploaded_file)
        else:
            response = requests.get(image_search_url)
            img_search = Image.open(BytesIO(response.content))
        c2.image(img_search, width=200, caption="Movies you would like to find")

        st.subheader("Results")
        data_load_state = st.empty()
        data_load_state.markdown("Searching results...")
        search_image_embedding = model.encode(img_search)
        target_by_image = str(search_image_embedding.tolist())
        result_by_image = movies.assign(
            cosine_distance=lambda t: cosine_distance(
                t["poster_emb"], vector(target_by_image)
            )
        ).order_by("cosine_distance")[:number_results]
        data_load_state.markdown(
            f"**{len(list(result_by_image))} Movies Found**: ... Printing Information..."
        )
        for row in result_by_image:
            container = st.container()
            col1, col2 = container.columns([1,4])
            with col1:
                st.image(f"https://image.tmdb.org/t/p/w500/{row['poster_path']}", width=200)
            with col2:
                st.markdown(
                   """<h4 style='text-align: center; color: black;'>{0}</h4>
                      <p style='text-align: left; color: black;'><b>Popularity:</b> {1} </p>
                      <p style='text-align: left; color: black;'><b>Genre:</b> {2} </p>
                      <p style='text-align: left; color: black;'><b>Release Date:</b> {3}</p>
                      <p style='text-align: left; color: black;'><b>Description:</b> {4} </p>
                   """.format(
                            row['title'], 
                            str(row['popularity']),
                            row["genres"], row["release_date"],
                            row["overview"]
                        ),
                    unsafe_allow_html=True)
        data_load_state.markdown(f"**{len(list(result_by_image))} Movies Found**")


if __name__ == "__main__":
    main()
