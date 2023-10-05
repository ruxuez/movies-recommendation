import streamlit as st

from streamlit_gallery.utils.db_helper import get_image_from_url
from sentence_transformers import SentenceTransformer, util

from multiprocessing import Pool
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

vector = gp.type_("vector", modifier=384)

movies = db.create_dataframe(table_name="movies_emb", schema="movies")

# First, we load the respective CLIP model
model = SentenceTransformer("all-MiniLM-L6-v2")

div_style = {
    "display": "flex",
    "justify-content": "center",
    "flex-wrap": "wrap",
}


def main():
    st.subheader("Instruction")
    st.markdown(
        "You can find the information of movies you want by entering the title. "
    )
    st.subheader("Description")
    title_search = st.text_input("Enter your movie title:", key="text")
    search_button = st.button("Search")

    if search_button:

        st.subheader("Movie Information")
        data_load_state = st.empty()
        data_load_state.markdown("Searching results...")
        lower_psql = gp.function("lower")
        result_by_text = movies.where(lambda t: lower_psql(t["title"]) == title_search.lower()).order_by("popularity", ascending=False)[:]
            
        data_load_state.markdown(
            f"**{len(list(result_by_text))} Movies Found**: ... Printing information..."
        )
        for row in result_by_text:
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


if __name__ == "__main__":
    main()
