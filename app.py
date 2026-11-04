import streamlit as st
import pandas as pd 
import tensorflow as tf
import pickle

#load preprocessors and modrl
model=tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl','rb') as file:
    le=pickle.load(file)
with open('onehot_encodr_country.pkl','rb') as file:
    ohe=pickle.load(file)
with open('standard_scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Customer Churn Prediction')

credit_score=st.number_input('Credit Score')
country=st.selectbox('Country',ohe.categories_[0])
gender=st.selectbox('Gender',le.classes_)
age=st.slider('Age',18,92)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
products_number=st.slider('No. of Products',1,5)
credit_card=st.selectbox('Has Credit Card',[0,1])
active_member=st.selectbox('Is a Active Number',[0,1])
estimated_salary=st.number_input('Estimated Salary')

#preparing input data
input={'credit_score':[credit_score],
'country':[country],
'gender':[le.transform([gender])[0]],
'age':[age],
'tenure':[tenure],
'balance':[balance],
'products_number':[products_number],
'credit_card':[credit_card],
'active_member':[active_member],
'estimated_salary':[estimated_salary]
}

input=pd.DataFrame(input)
cols=ohe.transform([[country]])
cols=pd.DataFrame(cols,columns=ohe.get_feature_names_out(),index=input.index)
input=input.drop(['country'],axis=1)
input=pd.concat([input,cols],axis=1)

input=scaler.transform(input)

#predict churnn
pred=model.predict(input)
predict_proba = model.predict(input)[0][0]

if predict_proba>0.75:
    st.write("The person is highly likely to churn")
elif predict_proba>0.5:
    st.write("The person is likely to churn")
else:
    st.write("The person is unlikely to churn")
st.write(f"The probabilty of person to churn is {predict_proba}")