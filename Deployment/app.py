import gradio as gr
from rag_module import (
    extract_text_from_pdf,
    parse_cv,
    generate_questions,
    answer_question
)
from feedback_module import process_video

parsed_cv_cache = {}

def generate_q_and_a(pdf_file):
    global parsed_cv_cache
    try:
        if not pdf_file:
            return "❌ Invalid file. Please upload a PDF.", "", "", "", ""

        # Extract text from the PDF file path
        file_path = pdf_file.name
        cv_chunks = extract_text_from_pdf(file_path)
        parsed_text = parse_cv(cv_chunks)
        parsed_cv_cache["cv"] = parsed_text

        # Generate interview questions
        questions_output = generate_questions(cv_chunks)

        # Extract technical and behavioral questions
        tech_qs, behave_qs = "", ""
        if "Technical:" in questions_output and "Behavioral:" in questions_output:
            tech_qs = questions_output.split("Technical:")[1].split("Behavioral:")[0].strip()
            behave_qs = questions_output.split("Behavioral:")[1].strip()
        else:
            return "⚠️ Failed to parse questions properly.", "", "", "", ""

        # Generate answers
        tech_q_lines = [line for line in tech_qs.splitlines() if line.strip() and line[0].isdigit()]
        behave_q_lines = [line for line in behave_qs.splitlines() if line.strip() and line[0].isdigit()]

        tech_ans = "\n\n".join([f"{q}\n{answer_question(q, parsed_cv_cache['cv'])}" for q in tech_q_lines])
        behave_ans = "\n\n".join([f"{q}\n{answer_question(q, parsed_cv_cache['cv'])}" for q in behave_q_lines])

        return parsed_text, "\n".join(tech_q_lines), "\n".join(behave_q_lines), tech_ans, behave_ans

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", ""

def answer_custom_question(q):
    global parsed_cv_cache
    if not parsed_cv_cache.get("cv"):
        return "⚠️ Please upload and parse a CV first."
    return answer_question(q, parsed_cv_cache["cv"])


def gradio_ui():
    with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as app:
        gr.Markdown("""
        <div style="text-align: center;">
            <h1 style="color: #3B82F6; font-size: 2.5em;">🤖 AI Interview Coach</h1>
            <p style="font-size: 1.2em;">Polish your responses. Master your presence.</p>
        </div>
        """)

        with gr.Tab("📄 Upload CV & Generate Questions"):
            with gr.Row():
                pdf_file = gr.File(label="Upload your CV (PDF only)", file_types=[".pdf"])
            with gr.Row():
                extracted_text = gr.Textbox(label="📄 Extracted CV Text", lines=8, interactive=False)
            with gr.Row():
                tech_q = gr.Textbox(label="🧠 Technical Questions", lines=6, interactive=False)
                behave_q = gr.Textbox(label="💬 Behavioral Questions", lines=6, interactive=False)
            with gr.Row():
                tech_ans = gr.Textbox(label="📘 Suggested Technical Answers", lines=8, interactive=False)
                behave_ans = gr.Textbox(label="📙 Suggested Behavioral Answers", lines=8, interactive=False)
            with gr.Row():
                gr.Button("🚀 Generate Q&A").click(
                    fn=generate_q_and_a,
                    inputs=pdf_file,
                    outputs=[extracted_text, tech_q, behave_q, tech_ans, behave_ans]
                )

        with gr.Tab("❓ Ask Custom Question"):
            with gr.Row():
                custom_q = gr.Textbox(label="🔍 Your custom question", placeholder="e.g. Tell me about a time you handled conflict.")
            with gr.Row():
                custom_a = gr.Textbox(label="📥 Answer from your CV & resources", lines=5, interactive=False)
            with gr.Row():
                gr.Button("💡 Get Answer").click(
                    fn=answer_custom_question,
                    inputs=custom_q,
                    outputs=custom_a
                )

        with gr.Tab("🎥 Video Interview Feedback"):
            with gr.Row():
                video_input = gr.Video(label="🎤 Upload or Record Interview")
            with gr.Row():
                transcript = gr.Textbox(label="📝 Transcript", lines=4, interactive=False)
            with gr.Row():
                behavior_fb = gr.Textbox(label="📉 Behavior Feedback", lines=5, interactive=False)
                answer_fb = gr.Textbox(label="📢 Answer Feedback", lines=5, interactive=False)
            with gr.Row():
                gr.Button("🔎 Analyze Video").click(
                    fn=process_video,
                    inputs=video_input,
                    outputs=[transcript, behavior_fb, answer_fb]
                )

        gr.Markdown("""
        ---
        <p style='text-align: center'>🚀 Made with ❤️ by <strong>IronIntellect</strong> — Elevate your interviews with AI</p>
        """)

    return app

if __name__ == "__main__":
    gradio_ui().launch(server_name="0.0.0.0", server_port=7860, share=True)
