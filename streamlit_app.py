import streamlit as st

from streamlit_gallery import apps, components
from streamlit_gallery.utils.page import page_group


def main():
    page = page_group("p")

    with st.sidebar:
        st.image("Greenplum_logo.png")
        st.title("🎥 Movies Recommendation System")

        with st.expander("✨ APPS", True):
            page.item("Movies gallery", apps.gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):
            page.item("Find by Movie", components.find_by_movie)
            page.item("Find by Description", components.find_by_text)
            page.item("Find by Poster", components.find_by_image)

    st.markdown(
        '<div style="text-align: right;"><sup><sub> <em>Built by VMware Data based on the <b>VMware Greenplum Data Warehouse</b></em> </sub></sup></div>',
        unsafe_allow_html=True,
    )

    page.show()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Gallery by VMware Data", page_icon="🎥", layout="wide"
    )
    main()
