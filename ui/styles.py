from config.config import  PRIMARY_COLOR, SECONDARY_COLOR, HIGHLIGHT_BG_COLOR

# CSS styles for the application
CSS = f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {PRIMARY_COLOR};
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {SECONDARY_COLOR};
        margin-top: 1rem;
    }}
    .highlight-text {{
        background-color: {HIGHLIGHT_BG_COLOR};
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }}
    .success-box {{
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }}
    .warning-box {{
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }}
    .error-box {{
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }}
    .info-box {{
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }}
</style>
"""


def inject_custom_css():
    """Inject custom CSS into the Streamlit app"""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def main_header(text):
    """Create a main header with custom styling"""
    import streamlit as st
    st.markdown(f'<p class="main-header">{text}</p>', unsafe_allow_html=True)


def sub_header(text):
    """Create a sub header with custom styling"""
    import streamlit as st
    st.markdown(f'<p class="sub-header">{text}</p>', unsafe_allow_html=True)


def highlight_text(text):
    """Create highlighted text with custom styling"""
    import streamlit as st
    st.markdown(f'<p class="highlight-text">{text}</p>', unsafe_allow_html=True)


def success_text(text):
    """Create success box with custom styling"""
    import streamlit as st
    st.markdown(f'<p class="success-box">{text}</p>', unsafe_allow_html=True)


def warning_text(text):
    """Create warning box with custom styling"""
    import streamlit as st
    st.markdown(f'<p class="warning-box">{text}</p>', unsafe_allow_html=True)


def error_text(text):
    """Create error box with custom styling"""
    import streamlit as st
    st.markdown(f'<p class="error-box">{text}</p>', unsafe_allow_html=True)


def info_text(text):
    """Create info box with custom styling"""
    import streamlit as st
    st.markdown(f'<p class="info-box">{text}</p>', unsafe_allow_html=True)
