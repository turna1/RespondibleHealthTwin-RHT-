import gradio as gr
import os
import re
import base64
import io
from PIL import Image
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Define slot templates
slot_templates = {
    "UQ": {"template": "Extract the main task and goal from the following input: {UserInput}. Output only the query between <UQ> and </UQ>.", "start_tag": "<UQ>", "end_tag": "</UQ>", "default": "Understand well-being need."},
    "CP": {"template": "Identify any personal. situational, or reelvant context from the following input: {UserInput}. Output the context between <CP> and </CP>. If none, output 'No context provided'.", "start_tag": "<CP>", "end_tag": "</CP>", "default": "Healthcare and well-being"},
    "J": {"template": "Set ethical guidelines or justification requirements for {UserInput}. Output between <J> and </J>.", "start_tag": "<J>", "end_tag": "</J>", "default": "Promote well-being."},
    "ROLE": {"template": "Determine the appropriate assistant role for responding to the following input:{UserInput}. Output between <ROLE> and </ROLE>.", "start_tag": "<ROLE>", "end_tag": "</ROLE>", "default": "well-being assistant"},
    "TONE": {"template": "Identify the suitable tone for responding to the following input: {UserInput}. Output between <TONE> and </TONE>.", "start_tag": "<TONE>", "end_tag": "</TONE>", "default": "supportive"},
    "FILT": {"template": "Specify any content filtering constraints for a safe and responsible response to the following input: {UserInput}. Output between <FILT> and </FILT>.", "start_tag": "<FILT>", "end_tag": "</FILT>", "default": "Ensure response is safe, respectful, and complies with ethical standards"},
    "FE": {"template": "Generate  synthetically 2-3 diverse complete few-shot example pairs (query and response) to guide the style and format relevant to the following input: {UserInput}. Output between <FE> and </FE>.", "start_tag": "<FE>", "end_tag": "</FE>", "default": "[Q: How can I improve my health? A: Maintain a healthy lifestyle]"}
}
# Global state
slots = {}
conversation = []

# Extract slot value
def extract_slot(response, slot):
    match = re.search(f"{re.escape(slot['start_tag'])}(.*?){re.escape(slot['end_tag'])}", response, re.DOTALL)
    return match.group(1).strip() if match else slot["default"]

# Call LLM
def call_llm(prompt):
    return client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512
    ).choices[0].message.content

# Convert image to data URL
def image_to_data_url(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

# Slot setup
def setup_from_input(user_input):
    global slots, conversation

    slots = {k: extract_slot(call_llm(v["template"].format(UserInput=user_input)), v) or v["default"] for k, v in slot_templates.items()}

    system_msg = f"You are a {slots['ROLE']}. Maintain a {slots['TONE']} tone and provide responses that are helpful and safe. Answer must follow these content constraints: {slots['FILT']}. Use these examples: {slots['FE']}to guide your style and format."
    user_msg = f"The user has submitted the following request: {slots['UQ']}\nContext: {slots['CP']}\n. Based on this query and context, provide a tailored, actionable response that fits the user’s lifestyle. Each recommendation should be supported by the following justification guideline: {slots['J']}"


    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    reply = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=conversation,
        temperature=0.5,
        max_tokens=512
    ).choices[0].message.content

    conversation.append({"role": "assistant", "content": reply})

    return tuple(map(str, [slots[k] for k in ["UQ", "CP", "J", "ROLE", "TONE", "FILT", "FE"]] + [user_msg, system_msg, reply]))

# Chat handler
def chat_with_image(user_text, user_image):
    if not conversation:
        return "Please initialize the assistant first."

    user_content = [{"type": "text", "text": user_text}]
    if user_image:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(user_image)}})

    conversation.append({"role": "user", "content": user_content})

    stream = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=conversation,
        temperature=0.7,
        max_tokens=1024,
        stream=True
    )

    reply = ""
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ""

    conversation.append({"role": "assistant", "content": reply})
    return reply

def save_slot_settings(uq_val, cp_val, j_val, role_val, tone_val, filt_val, fe_val,
                       user_prompt_val, sys_instr_val, assistant_output):
    import json

    data = {
        "UQ": uq_val,
        "CP": cp_val,
        "J": j_val,
        "ROLE": role_val,
        "TONE": tone_val,
        "FILT": filt_val,
        "FE": fe_val,
        "UserPrompt": user_prompt_val,
        "SystemInstruction": sys_instr_val,
        "InitialAssistantReply": assistant_output
    }

    with open("https://huggingface.co/spaces/Rahatara/WellebeingDT/blob/main/output.json", "w") as f:
        json.dump(data, f, indent=2)

    return "Settings and reply saved to output.json"

# UI
with gr.Blocks(theme="earneleh/paris") as demo:
    gr.Markdown("## 🧠 Responsible Well-Being Assistant")
    with gr.Tabs():
        with gr.Tab("1️⃣ Setup Assistant"):
            user_input = gr.Textbox(label="User Input", placeholder="e.g., How can I improve my sleep?")
            setup_btn = gr.Button("Generate Slots + Prompts")
            with gr.Accordion("Slots"):
                uq = gr.Textbox(label="User Query")
                cp = gr.Textbox(label="Context")
                j = gr.Textbox(label="Justification")
                role = gr.Textbox(label="Role")
                tone = gr.Textbox(label="Tone")
                filt = gr.Textbox(label="Filter")
                fe = gr.Textbox(label="Few-shot")
            user_prompt = gr.Textbox(label="User Prompt")
            sys_instr = gr.Textbox(label="System Instruction")
            setup_response = gr.Textbox(label="Initial Assistant Reply", lines=6)
            setup_btn.click(setup_from_input, inputs=user_input, outputs=[uq, cp, j, role, tone, filt, fe, user_prompt, sys_instr, setup_response])
            save_btn_tab1 = gr.Button("💾 Save Setup to output.json (Download)")
            save_status_tab1 = gr.Textbox(label="Save Status", interactive=False)
            download_btn_tab1 = gr.Button("📥 Download output.json")

            setup_btn.click(
                setup_from_input,
                inputs=[user_input],
                outputs=[uq, cp, j, role, tone, filt, fe, user_prompt, sys_instr, setup_response]
                )

            save_btn_tab1.click(
                fn=lambda uq_val, cp_val, j_val, role_val, tone_val, filt_val, fe_val, user_prompt_val, sys_instr_val, assistant_reply: (
                open("output.json", "w").write(json.dumps({
                "UQ": uq_val,
                "CP": cp_val,
                "J": j_val,
                "ROLE": role_val,
                "TONE": tone_val,
                "FILT": filt_val,
                "FE": fe_val,
                "UserPrompt": user_prompt_val,
                "SystemInstruction": sys_instr_val,
                "InitialAssistantReply": assistant_reply
            }, indent=2)) or "Saved to output.json"
            ),
            inputs=[uq, cp, j, role, tone, filt, fe, user_prompt, sys_instr, setup_response],
            outputs=[save_status_tab1]
            )

            download_btn_tab1.click(
                fn=lambda: "output.json",
                inputs=[],
                outputs=gr.File()
            )
        
        with gr.Tab("2️⃣ Multimodal Chat"):
            chat_input = gr.Textbox(label="Your Message")
            chat_image = gr.Image(type="pil", label="Upload Image (optional)")
            chat_btn = gr.Button("Send")
            chat_output = gr.Textbox(label="Assistant Reply", lines=6)
            chat_btn.click(chat_with_image, inputs=[chat_input, chat_image], outputs=chat_output)
        
        with gr.Tab("3️⃣ Digital Twin View"):
            from datetime import datetime
            import json

            gr.Markdown("""
            ### 🧬 Digital Twin Profile
            Adjust the sliders to simulate your well-being twin. Visual feedback and profile export included.
            """)

            well_being_score = gr.Slider(0, 100, value=75, step=1, label="Daily Well-Being Score")
            mood_factor = gr.Slider(0, 10, value=5, step=1, label="Mood Level")
            sleep_factor = gr.Slider(0, 10, value=7, step=1, label="Sleep Quality")
            energy_factor = gr.Slider(0, 10, value=6, step=1, label="Energy Level")
            mood_color = gr.ColorPicker(label="Mood Color", value="#90ee90")
            twin_btn = gr.Button("Generate Digital Twin View")
            download_btn = gr.Button("📥 Download Profile as JSON")
            twin_output = gr.HTML()

            def generate_digital_twin(score, color, mood, sleep, energy):
                goal = slots.get('UQ', 'N/A')
                context = slots.get('CP', 'N/A')
                tone = slots.get('TONE', 'N/A')

                description = f"""
                <div style="font-family:sans-serif;padding:1em;border-radius:12px;background-color:{color}20;border:2px solid {color};">
                    <h3>🌟 Your Personalized Well-being Digital Twin</h3>
                    <p><strong>Goal:</strong> {goal}</p>
                    <p><strong>Context:</strong> {context}</p>
                    <p><strong>Tone:</strong> {tone}</p>
                    <p><strong>Well-Being Score:</strong> {score}/100</p>
                    <div style="width:100%;height:20px;margin:1em 0;background-color:{color};"></div>
                    <p>🎯 Stay motivated and continue to make healthy choices! 💪</p>
                    <p><strong>Mood:</strong> {mood}/10 &nbsp; | &nbsp; <strong>Sleep:</strong> {sleep}/10 &nbsp; | &nbsp; <strong>Energy:</strong> {energy}/10</p>
                </div>
                """
                return description

            def download_profile(score, color, mood, sleep, energy):
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "goal": slots.get('UQ', 'N/A'),
                    "context": slots.get('CP', 'N/A'),
                    "tone": slots.get('TONE', 'N/A'),
                    "wellbeing_score": score,
                    "mood": mood,
                    "sleep": sleep,
                    "energy": energy,
                    "mood_color": color
                }
                with open("digital_twin_profile.json", "w") as f:
                    json.dump(data, f, indent=2)
                return "digital_twin_profile.json"

            twin_btn.click(
                fn=generate_digital_twin,
                inputs=[well_being_score, mood_color, mood_factor, sleep_factor, energy_factor],
                outputs=[twin_output]
            )

            download_btn.click(
                fn=download_profile,
                inputs=[well_being_score, mood_color, mood_factor, sleep_factor, energy_factor],
                outputs=gr.File()
            )
    
        with gr.Tab("4️⃣ Feedback Loop"):
            feedback_text = gr.Textbox(label="Optional Comment", placeholder="Tell us what could be better.")
            with gr.Row():
                like_btn = gr.Button("👍 Like")
                dislike_btn = gr.Button("👎 Dislike")
            updated_slots = gr.Textbox(label="Updated Slots After Feedback", lines=8)

            def update_slots_from_feedback(feedback):
                if not conversation:
                    return "Please initialize the assistant first."
                prompt = f"Extract semantic intent from: {feedback} You must Output in tag : <INTENT>...</INTENT>"
                feedback_intent = call_llm(prompt)
                intent_match = re.search(r"<INTENT>(.*?)</INTENT>", feedback_intent, re.DOTALL)
                intent = intent_match.group(1).strip() if intent_match else ""
                if intent:
                    slots["UQ"] += f" (adjusted for: {intent})"
                    slots["J"] += f" (in response to: {intent})"
                return str(slots)

            like_btn.click(fn=update_slots_from_feedback, inputs=[feedback_text], outputs=[updated_slots])
            dislike_btn.click(fn=update_slots_from_feedback, inputs=[feedback_text], outputs=[updated_slots])

if __name__ == "__main__":
    demo.launch()
