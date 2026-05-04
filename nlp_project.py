import gradio as gr
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# Pobranie modeli
nltk.download("punkt")
nltk.download("punkt_tab")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
zero_shot_model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

#TAB 1
def tokenize_text(text):
    # 1️⃣ Word-level (NLTK)
    word_tokens = word_tokenize(text)
    word_count = len(word_tokens)

    # 2️⃣ Subword (BERT)
    subword_tokens = bert_tokenizer.tokenize(text)
    subword_ids = bert_tokenizer.convert_tokens_to_ids(subword_tokens)
    subword_count = len(subword_tokens)

    return (
        " | ".join(word_tokens),
        word_count,
        " | ".join(subword_tokens),
        subword_count,
        " | ".join(map(str, subword_ids)),
    )

#Tab 2
def analyze_sentiment(text):
    result = sentiment_model(text)[0]

    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        positive_score = score
        negative_score = 1 - score
    else:
        negative_score = score
        positive_score = 1 - score

    return (
        f"{positive_score*100:.2f}%",
        f"{negative_score*100:.2f}%"
    )

#Tab 3
def compute_similarity(text1, text2):
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)

    cosine_score = F.cosine_similarity(
        embeddings[0],
        embeddings[1],
        dim=0
    )

    return f"{cosine_score.item():.4f}"

#Tab 4
def zero_shot_classify(text, labels):
    label_list = [label.strip() for label in labels.split(",")]

    result = zero_shot_model(
        text,
        candidate_labels=label_list
    )

    output = ""
    for label, score in zip(result["labels"], result["scores"]):
        output += f"{label}: {score*100:.2f}%\n"

    return output

#Tab 5
def summarize_text(text):
    input_tokens = summarizer.tokenizer(text, return_tensors="pt")
    input_length = input_tokens["input_ids"].shape[1]

    max_len = int(input_length * 0.6)
    min_len = int(input_length * 0.3)

    summary = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return summary


with gr.Blocks() as demo:
    with gr.Tab("Text Preprocessing - Tokenization"):
        gr.Markdown("## Compare Word-Level vs Subword Tokenization")

        text_input = gr.Textbox(
            value="Test input",
            label="Enter text",
            lines=4
        )

        tokenize_button = gr.Button("Tokenize")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Word-Level (NLTK)")
                word_output = gr.Textbox(label="Tokens")
                word_count_output = gr.Number(label="Token Count")

            with gr.Column():
                gr.Markdown("### Subword (BERT)")
                subword_output = gr.Textbox(label="Tokens")
                subword_count_output = gr.Number(label="Token Count")
                subword_ids_output = gr.Textbox(label="Token IDs")

        tokenize_button.click(
            tokenize_text,
            inputs=text_input,
            outputs=[
                word_output,
                word_count_output,
                subword_output,
                subword_count_output,
                subword_ids_output
            ]
        )

    with gr.Tab("Sentiment & Emotion Analysis"):
        gr.Markdown("## Sentiment Analysis using DistilBERT")

        sentiment_input = gr.Textbox(
            value="I absolutely love this NLP project!",
            label="Enter text",
            lines=4
        )

        sentiment_button = gr.Button("Analyze Sentiment")

        positive_output = gr.Textbox(label="Positive Confidence")
        negative_output = gr.Textbox(label="Negative Confidence")

        sentiment_button.click(
            analyze_sentiment,
            inputs=sentiment_input,
            outputs=[positive_output, negative_output]
        )

    with gr.Tab("Semantic Similarity"):
        gr.Markdown("## Semantic Similarity using Sentence Embeddings")

        text1_input = gr.Textbox(
            value="I love artificial intelligence.",
            label="Sentence 1",
            lines=3
        )

        text2_input = gr.Textbox(
            value="AI is amazing.",
            label="Sentence 2",
            lines=3
        )

        similarity_button = gr.Button("Compute Similarity")

        similarity_output = gr.Textbox(label="Cosine Similarity Score")

        similarity_button.click(
            compute_similarity,
            inputs=[text1_input, text2_input],
            outputs=similarity_output
        )

    with gr.Tab("Zero-Shot Topic Classification"):
        gr.Markdown("## Zero-Shot Topic Classification (BART-MNLI)")

        zs_text_input = gr.Textbox(
            value="Artificial intelligence is transforming modern healthcare.",
            label="Enter text",
            lines=4
        )

        zs_labels_input = gr.Textbox(
            value="Politics, Technology, Sports, Health",
            label="Enter categories (comma-separated)"
        )

        zs_button = gr.Button("Classify")

        zs_output = gr.Textbox(label="Classification Scores")

        zs_button.click(
            zero_shot_classify,
            inputs=[zs_text_input, zs_labels_input],
            outputs=zs_output
        )

    with gr.Tab("Text Summarization"):
        gr.Markdown("## Abstractive Text Summarization (BART)")

        sum_input = gr.Textbox(
            value="Artificial intelligence is transforming industries worldwide. "
                  "From healthcare to finance, AI systems are improving efficiency "
                  "and enabling new innovations. However, ethical concerns remain.",
            label="Enter text to summarize",
            lines=6
        )

        sum_button = gr.Button("Generate Summary")

        sum_output = gr.Textbox(label="Summary")

        sum_button.click(
            summarize_text,
            inputs=sum_input,
            outputs=sum_output
        )

demo.launch()