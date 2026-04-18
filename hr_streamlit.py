import streamlit as st
from hr_agent import app

st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 HR Policy Bot")
st.markdown("Ask me anything about company policies, leaves, salary, holidays, etc.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "web_session"
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke(
                {"question": prompt},
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
            answer = result["answer"]
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About")
    st.write("Answers from official HR policies only. No hallucination.")
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"session_{hash(str(st.session_state))}"
        st.rerun()