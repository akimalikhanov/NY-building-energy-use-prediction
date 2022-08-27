from gettext import npgettext
from re import L
import streamlit as st
from streamlit_folium import folium_static 
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import folium
from sklearn.preprocessing import MinMaxScaler


st.title('New York Building Energy Data')
st.write("EDA and Prediction of energy consumption in New York City.")

bar=st.sidebar.selectbox(
    "Sections",
    ("EDA", "Prediction")
)

def read_and_prep_data():
    data=pd.read_csv('data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016.csv')
    data = data.replace({'Not Available': np.nan})

    for col in list(data.columns):
        if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
            col or 'therms' in col or 'gal' in col or 'Score' in col):
            data[col] = data[col].astype(float)

    missing_cols=data.isnull().sum()
    missing_cols_percent=(missing_cols/len(data)*100).round(2)
    missing_df=pd.DataFrame({'No. of Missing Values':missing_cols, 
                            '% of Missing Values':missing_cols_percent})
    final=missing_df.sort_values(by=missing_df.columns[0], ascending=False)
    cols_to_delete=list(final[final.iloc[:,1]>50].index)
    data.drop(columns=cols_to_delete, inplace=True)

    q1, q3=data['Site EUI (kBtu/ft²)'].describe()[['25%', '75%']]
    iqr=q3-q1
    data=data[(data['Site EUI (kBtu/ft²)']>(q1-3*iqr)) & (data['Site EUI (kBtu/ft²)']<(q3+3*iqr))]
    return data


def score_hist():
    fig=px.histogram(df, x="ENERGY STAR Score", nbins=100,labels={
                    "ENERGY STAR Score": "Score"}).update_layout(yaxis_title="Number of Buildings")

    fig.update_traces(marker_line_width=1,marker_line_color="white")

    fig.update_layout(
        margin=dict(l=20, r=40, t=40, b=20))
    st.plotly_chart(fig)

def build_type_kdeplot(data, types):
    fig, ax=plt.subplots(figsize=(12,8))

    for t in types:
        subset=data[data['Largest Property Use Type']==t]
        sns.kdeplot(subset['ENERGY STAR Score'].dropna(), label=t)

    ax.set_xlabel('Energy Star Score', size=18)
    ax.set_ylabel('Density', size=18)
    ax.legend()
    st.pyplot(fig)

def build_folium_map(data):
    val_counts=pd.DataFrame(data['Borough'].value_counts())
    borogths_coors=data.groupby('Borough')[['Latitude', 'Longitude']].mean()
    borogths_score=pd.DataFrame(data.groupby('Borough')['ENERGY STAR Score'].mean())

    f = folium.Figure(width=700, height=700)
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11).add_to(f)

    for name, coors in zip(borogths_coors.index, borogths_coors.values):
        popup=f'''{name}<br>Coors: {coors}<br> Avg.Score: {borogths_score.loc[name].values[0]:.2f} 
        <br> No of Buildings: {val_counts.loc[name].values[0]}'''
        folium.Marker([coors[0], coors[1]], popup=popup, icon=folium.Icon(color='green', prefix='fa', icon='bolt')).add_to(m)
    folium_static(m)

def build_scatter(data, types):
    feats=data[data['Largest Property Use Type'].isin(types)]

    fig=px.scatter(feats, x='Site EUI (kBtu/ft²)', 
                y='ENERGY STAR Score', 
                color='Largest Property Use Type',
                width=1000, height=700)
                
    fig.update_layout(
        margin=dict(l=20, t=40, b=20),
    )
    st.plotly_chart(fig)
    

def prepare_data_for_model(data):
    X_train=pd.read_csv('data/testing_features.csv')
    X_train=X_train.loc[:,['Site EUI (kBtu/ft²)', \
        'Weather Normalized Site Electricity Intensity (kWh/ft²)',\
        'Weather Normalized Site Natural Gas Intensity (therms/ft²)',\
        'Year Built',\
        'Order',\
        'Property Id',\
        'Latitude',\
        'DOF Gross Floor Area',
        'Longitude',
        'Largest Property Use Type_Multifamily Housing']]
    sc=MinMaxScaler().fit(X_train)
    data=sc.transform(data)
    return data

def predict(data):
    model=pickle.load(open('models/reduced_gbtree.sav', 'rb'))
    y_pred=model.predict(data)
    return y_pred[0]


df=read_and_prep_data()


if bar=='EDA':
    df=read_and_prep_data()
    types=df.dropna(subset=['ENERGY STAR Score'])
    types=types['Largest Property Use Type'].value_counts()
    types=list(types[types.values>100].index) # use only types with 100+ observations in dataset
    with st.container():
        st.header("Plot One: Energy Star Score Distribution")
        score_hist()

    with st.expander("About Energy Star Score Distribution"):
        st.write('''
        Surprisingly, there are two peaks and maximum and minimum values: 0 and 100. 
        The problem might be in data collection. Turns out, this score is based on sefl-reported energy usage. 
        Hence, some building owners might just lower the actual power usage to artificially boost the score of their building.
        ''')
    
    with st.container():
        st.header('Plot Two: Density Plot of Energy Star Scores by Building Type')
        build_type_kdeplot(df, types)

    with st.expander("About Density Plot by Building Type"):
        st.write('''
        From here we can see that building type does have some effect on the score. 
        E.g., offices tend to have higher score  compared to other building types.
        Contrary, hotels on average have the lowest score.
        ''')

    with st.container():
        st.header('Plot Three: NYC Map with Average Score and No. of Buildings by Borough')
        build_folium_map(df)

    with st.expander("About NYC Map"):
        st.write('''
        Manhattan has the largest number of buildings (5176) while Staten Island has only 159. 
        The highest average score is in Brooklyn (63), the lowest in Bronx (58).
        ''')

    with st.container():
        st.header('Plot Four: Energy Star Score vs Site EUI')
        build_scatter(df, types)

    with st.expander("About Energy Star Score vs Site EUI"):
        st.write('''
        There is a clear negative relationship between the Site EUI and the score. 
        The relationship is not perfectly linear but it does look like this feature can help to predict the score.
        ''')

else:
    x_dict={'Site EUI (kBtu/ft²)': None,
            'Weather Normalized Site Electricity Intensity (kWh/ft²)':None,
            'Weather Normalized Site Natural Gas Intensity (therms/ft²)':None,
            'Year Built':None,
            'Order':None,
            'Property Id':None,
            'Latitude':None,
            'DOF Gross Floor Area':None,
            'Longitude':None,
            'Largest Property Use Type_Multifamily Housing':None}

    # if st.button('Submit'):
    st.info('Fill in all boxes and press Submit button to get a predicted score')
    with st.form("my_form"):
        col1, col2=st.columns(2)
        with col1:
                x_dict['Site EUI (kBtu/ft²)']=st.number_input('Site EUI (kBtu/ft²)')
                x_dict['Weather Normalized Site Electricity Intensity (kWh/ft²)']=st.number_input('Weather Normalized Site Electricity Intensity (kWh/ft²)')
                x_dict['Weather Normalized Site Natural Gas Intensity (therms/ft²)']=st.number_input('Weather Normalized Site Natural Gas Intensity (therms/ft²)')
                x_dict['Year Built']=st.number_input('Year Built')
                x_dict['Order']=st.number_input('Order')
        with col2:
            x_dict['Property Id']=st.number_input('Property Id')
            x_dict['Latitude']=st.number_input('Latitude')
            x_dict['DOF Gross Floor Area']=st.number_input('DOF Gross Floor Area')
            x_dict['Longitude']=st.number_input('Longitude')
            x_dict['Largest Property Use Type_Multifamily Housing']=st.number_input('Largest Property Use Type_Multifamily Housing')
        
        submitted = st.form_submit_button("Submit")

        if submitted:
            d=prepare_data_for_model(np.array(list(x_dict.values())).reshape(1,-1))
            st.success(f'The Predicted Energy Star Score is: {predict(d):.2f}')


