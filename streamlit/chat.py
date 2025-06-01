import streamlit as st


# page config
st.set_page_config(page_title="소득세 챗봇", page_icon=":robot_face:", layout="wide")

st.title(":robot_face: 소득세 챗봇")
st.caption("소득세 관련 질문을 해보세요!")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print(f"before: {st.session_state.message_list}")
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"]) # 쌓인 질문을 모두 화면에 그린다.

if user_question := st.chat_input(placeholder="소득세 관련 질문을 해보세요..."):
    with st.chat_message("user"):
        st.write(user_question) # 질문 남기기
    st.session_state.message_list.append({"role": "user", "content": user_question})
print(f"after: {st.session_state.message_list}")
