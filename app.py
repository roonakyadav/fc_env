import gradio as gr

def run_agent():
    return "FC Env PPO Agent is running"

demo = gr.Interface(
    fn=run_agent,
    inputs=[],
    outputs="text"
)

demo.launch()
