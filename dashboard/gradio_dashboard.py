import pandas as pd
import numpy  as np
from dotenv import load_dotenv


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from langchain_chroma import Chroma
import os

import gradio as gr
load_dotenv()

df = pd.read_csv('data/books_with_emotions.csv')
# thumbnails of different books have random sizes, so we will set all of them to max size
df['large_thumbnail'] = df['thumbnail'] + "&fife=w800"

# replace missing thumbnails with cover not found
df["large_thumbnail"] = np.where(
    df["large_thumbnail"].isna(),
    "../assets/cover-not-found.jpg",
    df["large_thumbnail"],
)

embeddings = JinaEmbeddings(
    jina_api_key=os.getenv('JINA_API_KEY'),
    model_name="jina-embeddings-v2-base-en"
)

raw_documents = TextLoader('text_files/tagged_description.txt',encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0,separator="\n")
documents = text_splitter.split_documents(raw_documents)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db/chroma_db_jina"  
)

# a function that retrieves books based on a user query and tone,category filters
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = vectorstore.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = df[df["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

# function to recommend books
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(df["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# dashboard design
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
    dashboard.launch()