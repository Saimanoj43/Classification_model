###### Libraries #######

# Base Libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA

# Deployment Library
import streamlit as st

# Model Pickled File Library
import joblib

############# Data File ###########

data = pd.read_csv('Hotel_booking.csv')


data = data.dropna(axis=0).reset_index(drop=True)

########### Loading Trained Model Files ########
model = joblib.load("Hotel_booking.pkl")
model_ohe = joblib.load("Hotel_booking_ohe.pkl")
model_sc = joblib.load("Hotel_booking_sc.pkl")
model_pca = joblib.load("Hotel_booking_pca.pkl")

########## UI Code #############

# Ref: https://docs.streamlit.io/library/api-reference

# Title
st.title(" Estimation of hotel booking cancellation based on the Given Person Details:")

# Image
with st.columns(5)[1]:
    st.image("https://png.pngtree.com/png-clipart/20230921/original/pngtree-hotel-booking-word-concepts-banner-trip-planning-amenities-reviews-vector-png-image_12474719.png", width=400)

# Description
st.write("""Built a Predictive model in Machine Learning to estimate the hotel booking cancellation of a person can get.
         Sample Data taken as below shown.
""")

# Data Display
del data['Unnamed: 0']
del data['company']
del data['is_canceled']
del data["agent"]
del data["market_segment"]
del data['arrival_date_week_number']
del data["customer_type"]

st.dataframe(data.head())
st.write("From the above data , Price is the prediction variable")

###### Taking User Inputs #########
st.subheader("Enter Below Details to Get the Estimation Prices:")

col1, col2, col3,col4 = st.columns(4) # value inside brace defines the number of splits
col5, col6,col7,col8 = st.columns(4)
col9, col11,col25 = st.columns(3)
col10,col12,col13 = st.columns(3)
col14,col15,col16,col26= st.columns(4)
col17, col18, col19,col20 = st.columns(4) # value inside brace defines the number of splits
col21, col22, col23,col24 = st.columns(4)



with col1:
    hotel = st.selectbox("Enter Hotel Type:",data.hotel.unique())
    st.write(hotel)

with col2:
    lead_time = st.number_input("Enter lead_time (in days):")
    st.write(lead_time)


with col3:
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    arrival_date_month = st.selectbox("Enter month of arrival :",months)
    st.write(arrival_date_month)

with col4:
    arrival_date_day_of_month = st.number_input("Enter date of arrival :",min_value=0)
    st.write(arrival_date_day_of_month)

with col5:
    stays_in_weekend_nights = st.number_input("Enter number of weekend nights :",min_value=0)
    st.write(stays_in_weekend_nights)

with col6:
    stays_in_week_nights = st.number_input("Enter number of weekday nights :",min_value=0)
    st.write(stays_in_week_nights)

with col7:
    adults = st.number_input("Enter number of adults :",min_value=0)
    st.write(adults)

with col8:
    children = st.number_input("Enter number of children :",min_value=0)
    st.write(children)

with col9:
    babies = st.number_input("Enter number of babies :",min_value=0)
    st.write(babies)

with col10:
    st.text("BB : Bed & Breakfast")
    st.text("HB : Half board (breakfast and one other meal-usually dinner)")
    st.text("FB : Full board (breakfast, lunch, and dinner)")
    st.text("SC : no meal package")
    meal = st.selectbox("Enter meal options from options *BB : Bed & Breakfast:",data.meal.unique())
    st.text("*BB : Bed & Breakfast")
    st.write(meal)

with col11:
    country = st.selectbox("Enter country from above options :",data.country.unique())
    st.write(country)

with col12:
    st.text("TA : Travel agents")
    st.text("TO : Tour Operators")
    distribution_channel = st.selectbox("Enter distribution from above options :",data.distribution_channel.unique())
    st.write(distribution_channel)

with col13:
    st.text("0 : New visiter ")
    st.text("1 : Already visited user")
    is_repeated_guest = st.selectbox("Enter repeated customer or not from above options :",data.is_repeated_guest.unique())
    st.write(is_repeated_guest)


with col14:
    previous_cancellations = st.number_input("Enter number of previous cancellations : ",min_value=0)
    st.write(previous_cancellations)

with col15:
    previous_bookings_not_canceled = st.number_input("Enter number of previous bookings not canceled : ",min_value=0)
    st.write(previous_bookings_not_canceled)

with col16:
    reserved_room_type = st.selectbox("Enter reserved room type : ",data.reserved_room_type.unique())
    st.write(reserved_room_type)

with col17:
    assigned_room_type = st.selectbox("Enter assigned room type from above options : ",data.assigned_room_type.unique())
    st.write(assigned_room_type)

with col18:
    booking_changes = st.number_input("Enter number of times bookings changed : ",min_value=0)
    st.write(booking_changes)

with col19:
    deposit_type = st.selectbox("Enter deposit details from above options :",data.deposit_type.unique())
    st.write(deposit_type)

with col20:
    days_in_waiting_list = st.number_input("Enter no.of days in waiting list to get confirmation : ",min_value=0)
    st.write(days_in_waiting_list)

with col21:
    st.write("Average daily rate")
    adr = st.number_input("Enter adr details from above options :")
    st.write(adr)

with col22:
    required_car_parking_spaces = st.number_input("Enter no.of parking spaces required :",min_value=0)
    st.write(required_car_parking_spaces)

with col23:
    total_of_special_requests = st.number_input("Enter no.of special_requests from customer :",min_value=0)
    st.write(total_of_special_requests)

with col24:
    reservation_status = st.selectbox("Enter reservation status details from above options :",data.reservation_status.unique())
    st.write(reservation_status)

with col25:
    reservation_status_date = st.date_input("Enter Last status checked by customer (in days) :")
    st.write(reservation_status_date)

with col26:
    arrival_date_year = st.number_input("Enter Arrival year :",min_value=0)
    st.write(arrival_date_year)

###### Predictions #########

if st.button("Booking Status"):
    st.write("Data Given:")
    values = [hotel,lead_time,arrival_date_month,arrival_date_day_of_month,stays_in_weekend_nights,
                stays_in_week_nights,adults,children,babies,meal,country,distribution_channel,
                is_repeated_guest,previous_cancellations,previous_bookings_not_canceled,
                reserved_room_type,assigned_room_type,booking_changes,deposit_type,days_in_waiting_list,
                adr,required_car_parking_spaces,total_of_special_requests,reservation_status,reservation_status_date,arrival_date_year]
    record =  pd.DataFrame([values],
                           columns = ['hotel','lead_time','arrival_date_month','arrival_date_day_of_month','stays_in_weekend_nights',
                                'stays_in_week_nights','adults','children','babies','meal','country','distribution_channel',
                                'is_repeated_guest','previous_cancellations','previous_bookings_not_canceled',
                                'reserved_room_type','assigned_room_type','booking_changes','deposit_type','days_in_waiting_list',
                                'adr','required_car_parking_spaces','total_of_special_requests','reservation_status','reservation_status_date','arrival_date_year'])
    
    st.dataframe(record)
    arrival_date_year = int(arrival_date_year)


    LastStatusUpdate = []

    # Loop to calculate LastStatusUpdate
    for i in range(len(record)):
        days = 0
        s = ''
        s = s + str(record['arrival_date_day_of_month'].iloc[i]) + " " + str(record['arrival_date_month'].iloc[i]) + ", " + str(record['arrival_date_year'].iloc[i])

        # Remove any trailing ".0" from the date string
        if s.endswith('.0'):
            s = s[:-2]

        d = datetime.strptime(s, '%d %B, %Y').date()  # Convert to datetime.date object
        x = record['reservation_status_date'].iloc[i]

        # Assuming x is already a datetime object, you don't need to convert it
        days = (x - d).days  # Calculate the difference in days
        LastStatusUpdate.append(days)  # Append days to LastStatusUpdate list

    # Assign 'LastStatusUpdate' column after the loop ends
    record['LastStatusUpdate'] = LastStatusUpdate

    # Remove unnecessary columns
    record.drop(['reservation_status_date', 'arrival_date_year'], axis=1, inplace=True)


    for col in record.columns:
        if record[col].dtype=='object':
            record[col] = record[col].str.lower()

    record.hotel.replace({'resort hotel':0,'city hotel':1},inplace=True)
    record.reserved_room_type.replace({'l':0,'h':1,'c':2,'b':3,'g':4,'f':5,'e':6,'d':7,'a':8},inplace=True)
    record.assigned_room_type.replace({'k':0,'i':1,'l':2,'h':3,'c':4,'b':5,'g':6,'f':7,'e':8,'d':9,'a':10},inplace=True)
    record.reservation_status.replace({'check-out':1, 'canceled':2, 'no-show':0},inplace=True)


    ohedata1 = model_ohe.transform(record.iloc[:,[2,9,10,11,18]]).toarray()
    ohedata1 = pd.DataFrame(ohedata1,columns=model_ohe.get_feature_names_out())
    record=pd.concat([record.iloc[:,0:2],record.iloc[:,3:9],record.iloc[:,12:18],record.iloc[:,19:],ohedata1],axis=1)
    

    record.iloc[:,[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]] = model_sc.transform(record.iloc[:,[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]])
    

    x_pca = model_pca.transform(record) 
    x_pca = pd.DataFrame(x_pca[:, :14])
   


    # st.dataframe(record)
    is_cancelled_status = model.predict(x_pca)[0]
    st.subheader("Booking Status:")
    
    if is_cancelled_status==0:
        st.subheader("Booking is not cancelled")
    else:
        st.subheader("Booking is cancelled")

