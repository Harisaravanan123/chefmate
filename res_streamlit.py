import folium.map
import streamlit as st
import pickle
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import OneHotEncoder
import google.generativeai as genai
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import pandas as pd



# load the model and other preprocessing techniques
# with open('minikmeans - Copy.pkl','rb')as model_file:
#     model=pickle.load(model_file)
# with open('encoder.pkl','rb')as encoder_file:
#     ohe=pickle.load(encoder_file)    
# with open('pca - Copy.pkl','rb')as pca_file:
#     pca=pickle.load(pca_file)  
# load the file
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
connection=psycopg2.connect(
    host='database-1.crum2kge4eek.ap-south-1.rds.amazonaws.com',
    port=5432,
    user='postgres',
    password='Hari1777'
)
connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
writer=connection.cursor()
writer.execute(''' SELECT * from "Restaurant_data" ''')
tables=writer.fetchall()
column_names=[desc[0] for desc in writer.description]
df=pd.DataFrame(tables,columns=column_names)

with open('minikmeans.pkl','rb')as model_file:
    model=pickle.load(model_file)
with open('encoder.pkl','rb')as encoder_file:
    ohe=pickle.load(encoder_file)    
with open('pca.pkl','rb')as pca_file:
    pca=pickle.load(pca_file)  

st.set_page_config(page_title='Restaurant Recommendation System and Chefbot',page_icon='robot_face',layout='wide')


page=st.sidebar.radio("select a page:",("Restaurant Recommendation","ChefbotGPT"))  
if page=="Restaurant Recommendation":



# set the title
    title=st.title("RESTAURANT RECOMMENDATION SYSTEM")
    
    st.sidebar.header("Filter Options")
    Cuisines=df['Cuisines'].unique()
    selected_Cuisines=st.sidebar.selectbox("SELECT THE CUISINES:",options=Cuisines)
    
    def get_recommendation(selected_cuisines):
        filtered_df=df[df["Cuisines"]==selected_Cuisines ]
        if filtered_df.empty:
            return pd.DataFrame()
        cuisine_encoded=ohe.transform([[selected_Cuisines]])
        cusine_pca=pca.transform(cuisine_encoded)
        cluster=model.predict(cusine_pca)
        restaurant_recommendation=filtered_df[filtered_df['Cluster']==cluster[0]]
        restaurant_recommendation=restaurant_recommendation.sort_values(by='Rating', ascending=False)
        restaurant_recommendation=restaurant_recommendation.drop_duplicates(subset='Name')
        return restaurant_recommendation
    if selected_Cuisines:
        
        recommendations=get_recommendation(selected_Cuisines)   
        if not recommendations.empty:
            st.subheader("**RECOMMENDED_RESTAURANTS**")  
            st.dataframe(recommendations[['Name','Location','City','Rating']])
            st.subheader("**RESTAURANT LOCATION**")
            m=folium.Map(location=[recommendations['latitude'].mean(),recommendations['longitude'].mean()])
            for idx,row in recommendations.iterrows():
                folium.Marker(
                    location=[row['latitude'],row['longitude']],
                    popup=f"{row['Name']}  -  {row['Location']}",
                    icon=folium.Icon(color='blue')

                ).add_to(m)
            st_folium(m,width=1000,height=700)    
        else:
            st.write('no restaurants recommended')

# create a  chatbot  for cooking instructions only:
# st.sidebar.radio("chatbot")
if page=="ChefbotGPT":
    
    st.title("cooking instruction chatbot")
    
    st.header("Ask me anything about cooking")
    genai.configure(api_key="AIzaSyAB1BCVmzol8848PmJThLPjH8Rrcvu_ADE")
    model=genai.GenerativeModel("gemini-1.5-flash")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    user_input=st.text_input("Enter your cooking related questions:")
    if user_input:
        chat=model.start_chat(
            history=[
                {"role":"user",'parts':'hello'},
                {"role":"model","parts":"Give answers only for cooking related questions"},
            ]
        )
        response=chat.send_message(user_input)
        if response and response.candidates:
            bot_response=response.candidates[0].content.parts[0].text
        else:
            bot_response="Sorry I could not able to generate a response"    
        st.session_state.chat_history.append({'user':user_input,'bot':bot_response})
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.write(f"**You:**{chat['user']}")  
            st.write(f"**bot:**{chat['bot']}")  
        





