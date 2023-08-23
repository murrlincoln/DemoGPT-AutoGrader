# Prompt: Create a versatile auto-grader application capable of evaluating different types of assignments—ranging from code to essays to mathematical problems. The auto-grader should analyze submissions for key metrics such as accuracy, coherence, effectiveness, and overall quality. 
#**User Customization:**
#- Enable the user to input a custom prompt that outlines the assignment criteria and objectives. This will serve as a guideline for the AI to evaluate the submission against.
#**Grading Metrics:**
#- For code, check for syntactical errors, code readability, and efficiency.
#- For essays, evaluate grammar, coherence, argument strength, and evidence.
#- For mathematical problems, assess the correctness of solutions, clarity in steps, and logical reasoning.
# Provide one of many pre-generated responses and grades for the work.
# **Output:**
# The auto-grader should display the grade and an assessment.

import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document

def load_submission(submission_path):
    from langchain.document_loaders import UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(submission_path, mode="elements", strategy="fast")
    submission_doc = loader.load()
    return submission_doc

def assignmentEvaluator(assignment_type,criteria_objectives,submission_string):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an AI assistant tasked with evaluating a submission based on the type of assignment, criteria, and objectives."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """The assignment type is '{assignment_type}', and the criteria and objectives are '{criteria_objectives}'. Please evaluate the following submission: '{submission_string}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(assignment_type=assignment_type, criteria_objectives=criteria_objectives, submission_string=submission_string)
    return result

def gradeEvaluator(evaluation):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an AI assistant tasked with evaluating and grading a submission."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please evaluate the following submission: '{evaluation}'. Provide a grade and feedback."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(evaluation=evaluation)
    return result

def display_grade_assessment(grade_assessment):
    if grade_assessment:
        st.markdown(f"**Grade and Assessment:** {grade_assessment}")
    else:
        st.markdown("No grade and assessment available.")

st.title('Auto-Grader')

assignment_type = st.text_input('Enter the type of assignment')
criteria_objectives = st.text_area('Enter the assignment criteria and objectives')
uploaded_file = st.file_uploader("Upload Your Assignment", type=["txt", "docx", "pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        submission_path = temp_file.name
        st.session_state['submission_path'] = submission_path
else:
    submission_path = ""

if 'submission_path' in st.session_state and st.session_state['submission_path']:
    submission_doc = load_submission(st.session_state['submission_path'])
    submission_string = "".join([doc.page_content for doc in submission_doc])
else:
    submission_string = ""

if st.button('Evaluate'):
    if assignment_type and criteria_objectives and submission_string:
        evaluation = assignmentEvaluator(assignment_type,criteria_objectives,submission_string)
    else:
        evaluation = ""

    if evaluation:
        grade_assessment = gradeEvaluator(evaluation)
    else:
        grade_assessment = ""

    display_grade_assessment(grade_assessment)
