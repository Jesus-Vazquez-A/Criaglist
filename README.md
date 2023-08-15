# Generation of Competitive Prices in the Used Car Market through Neural Networks.

In the dynamic used car market, determining a competitive price is essential to attract buyers and maximize sales. However, this task becomes increasingly challenging due to the large number of influencing variables and the lack of structured information. In this context, this proyect proposed a solution based on the use of neural networks to generate competitive prices for used cars.

The methodology was based on the collection of historical used car sales data, including characteristics such as brand, model, year of manufacture, mileage, vehicle specifications, etc. We used vehicles with a manufacturing date greater than or equal to the year 2000, as they are the most commonly found cars on the road today. This data was used to train a deep neural network, specifically designed to understand complex patterns and non-linear relationships between variables. The neural network was able to autonomously learn the relationships between the car's characteristics and its final sale price.

Before proceeding with the creation of the model, we had to clean the data, there were lost and erroneous data, we had to correct the name of the models, since they had syntax errors. As well as carrying out transformations of variables, as is the case of applying square root extraction for the odometer variable so that the extreme values are closer to the average value.

We tested various ML models such as linear regression such as XGBoost, the first model got decent results and the second got very interesting metrics. However, the latter has a disadvantage in that it is not good for estimating the price of cars for those with a manufacturing date greater than 2024, something that the neural network does very well.

Separating the data into training, test and validation in the three data groups obtained very good metrics, in addition to applying methods to prevent excessive overfitting, such as deactivating a certain percentage of neurons in order to search for several alternative patterns and applying a significantly lower learning rate.

RMSE Train: 3505.004
RMSE Test:  3690.414
RMSE Val:   3690.170


Finally we use this model to use it in a Streamlit application so that several users can use it in real time.



