import streamlit as st
import pickle

# Spam messages are basically repeated messages

import numpy as np
import pandas as pd 
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# import base64

# # Background image from computer using downloaded image
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('pic.png')    



# Background image from internet using url
def add_bg_from_url():
    st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url("https://assets.website-files.com/5af0be84599b0d3f7a3f67c2/5af0c33e187a2d73fc328078_4-tips-for-a-successful-email-newsletter.jpg");
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-size: 2500px 1000px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url() 



df_sms = pd.read_csv('spam.csv', encoding = 'latin-1')
df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
df_sms = df_sms.rename(columns={"v1" : "label", "v2" : "sms"})
df_sms['length'] = df_sms['sms'].apply(len)
df_sms.loc[:,'label'] = df_sms.label.map({'ham' : 0, 'spam' : 1})

# Splitting the dataset into training set and testing set along x and y
# And then we are labelling the data inside
X_train, X_test, y_train, y_test = train_test_split(
    df_sms['sms'],
    df_sms['label'], test_size = 0.20,
    random_state = 1
)

count_vector = CountVectorizer() # counts no. of words entered by the user
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# Naive Bayes is a classifier and Multinomial is a type of it
naive_bayes = MultinomialNB() 
naive_bayes.fit(training_data, y_train)

MultinomialNB(alpha = 1.0, class_prior = None, fit_prior = True)

predictions = naive_bayes.predict(testing_data)


#model=pickle.load('spam msg classifier.py','rb')

st.title("Spam Message Classifier")

st.write("Want to upload a file and classify it ?")

agree = st.checkbox('I agree')

if agree:

    st.write("Great! Let's upload the file(s).")
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

    # opening one file at a time to process
    for uploaded_file in uploaded_files:

        # Receiving the data given by the user from the webpage
        bytes_data = uploaded_file.read()

        file_name="\nFile uploaded: "+uploaded_file.name+"\n"
        st.header(file_name)
    
        with open("write file.txt","w") as r:

            # transforming the bytes to text
            txt_data = bytes_data.decode('UTF-8')
            r.write(txt_data)
    
        text=[] # line list
        nsd,sd="",""

        # Removing the lines in the list without "\n"
        with open("write file.txt","r") as r:
            fr=r.readline()
        
            while fr:
                if fr != "\n":
                    text.append(fr)
                fr=r.readline()
    
        # Adding the content of the list to the usable file
        with open("write file.txt","w") as w:
            for i in text:
                w.writelines(i)

        fo=open("write file.txt","r")
        input=fo.readline() # reading lines from the given file

        while input:

            inp = np.array(input) # converting to an array
            inp = np.reshape(inp, (1, -1)) # Restructuring the output that we get from the program
            inp_conv = count_vector.transform(inp.ravel()) # counts frequency of word
            result = naive_bayes.predict(inp_conv) # 0 or 1

        
            for element in result : # checking if the message type is present in the csv file 
                if result[0] == 0 : # determining spam or not
                    nsd=nsd+input
                else :
                    sd=sd+input
            
            input=fo.readline()
    

        # Download buttons for downloading the processed file

        st.subheader("All Spams are removed!")
        st.write("Click on the button below to download your File after spam has been removed")
        st.download_button('Download Your File', nsd)  # Defaults to 'text/plain'  
    
        st.subheader("We have stored all the spam data in a file")
        st.write("Click on the button below to download the file if you want to take a look.")
        st.download_button('Download', sd)  # Defaults to 'text/plain'
    
            

    st.write('Thanks for opting us!')  

else:
    st.write("Want to check a message is spam or not ?")
    input = st.text_area("Enter the message: ")

    if st.button("Predict"):

        # 1. Preprocess
        # 2. Vectorize
        # 3. Predict
        # 4. Display


        #con_inp = input('Enter a message : ') # user input
        inp = np.array(input)
        inp = np.reshape(inp, (1, -1)) # Restructuring the output that we get from the program
        inp_conv = count_vector.transform(inp.ravel())
        result = naive_bayes.predict(inp_conv)

        for element in result : # checking if the message type is present in the csv file 
            if result[0] == 0 : # determining spam or not
                #print('It is not a spam')
                st.header("It is not a Spam")
            else :
                #print('It is a spam')
                st.header("It is a Spam")


