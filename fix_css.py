f=open('streamlit_app.py')
c=f.read()
f.close()
css = 'st.markdown("""\n<style>\n    .main .block-container { padding-top: 2rem; }\n    h1 { color: #001f3f; }\n    h2 { color: #6f42c1; }\n    .stButton>button { background: linear-gradient(135deg, #6f42c1, #5a32a3); color: white; border: none; border-radius: 10px; }\n</style>\n""", unsafe_allow_html=True)\n'
c = c.replace('st.title', css + 'st.title', 1)
f=open('streamlit_app.py','w')
f.write(c)
f.close()
print('Done')
