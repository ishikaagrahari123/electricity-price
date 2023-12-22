% Load the dataset 
filePath = 'C:\Users\Shourya\Downloads\merged_dataset.csv';
data = readtable(filePath);


%corresponding data 

% var1 = data.date;
% var2 = data.time;
% var3 = data.city_name;
% var4 = data.temp;
% var5 = data.temp_min;
% var6 = data.temp_max;
% var7 = data.pressure;
% var7 = data.humidity;
% var8 = data.wind_speed;
% var9 = data.wind_deg;
% var10 = data.rain_1h;
% var11 = data.rain_3h;
% var12 = data.snow_3h;
% var13 = data.clouds_all;
% var14 = data.weather_id;
% var15 = data.weather_main;
% var16 = data.weather_description;
% var17 = data.weather_icon;
% var18 = data.date;
% var19 = data.time
% var10 = data.generation_biomass;
% var21 = data.generation_fossil_brown_coal_lignite;
% var22 = data.generation_fossil_coal_derived_gas;
% var23 = data.generation_fossil_gas;
% var24 = data.generation_fossil_hard_coal;
% var25 = data.generation_fossil_oil;
% var26 = data.generation_fossil_oil_shale;
% var27 = data.generation_fossil_peat;
% var28 = data.generation_geothermal;
% var29 = data.generation_hydro_pumped_storage_aggregated;
% var30 = data.generation_hydro_pumped_storage_consumption;
% var31 = data.generation_hydro_run_of_river_and_poundage;
% var32 = data.generation_hydro_water_reservoir;
% var33 = data.generation_marine;
% var34 = data.generation_nuclear;
% var35 = data.generation_other;
% var36 = data.generation_other_renewable;
% var37 = data.generation_solar;
% var38 = data.generation_waste;
% var39 = data.generation_wind_offshore;
% var40 = data.generation_wind_onshore;
% var41 = data.forecast_solar_day_ahead;
% var42 = data.forecast_wind_offshore_eday_ahead;
% var43 = data.forecast_wind_onshore_day_ahead;
% var44 = data.total_load_forecast;
% var45 = data.total_load_actual;
% var47 = data.price_day_ahead;
% var48 = data.price_actual;

% Plot electricity prices over time
figure;
subplot(2, 1, 1);
plot(data.Var1, data.Var48);
xlabel('Timestamp');
ylabel('Price');
title('Electricity Prices Over Time');

% Plot selected features over time
subplot(2, 1, 2);
plot(data.Var1, data.Var4, 'r', ...  % Temperature in red
     data.Var1, data.Var8, 'b');     % Humidity in blue
xlabel('Timestamp');
title('Weather Features Over Time');
legend('Temperature',  'Humidity');

% Create a histogram of electricity prices
figure;
histogram(data.Var48, 20); % You can adjust the number of bins (e.g., 20) as needed
xlabel('Price');
ylabel('Frequency');
title('Electricity Prices Histogram');

% Scatter plot of temperature vs. electricity prices
figure;
scatter(data.Var4, data.Var48);
xlabel('Temperature');
ylabel('Price');
title('Scatter Plot: Temperature vs. Electricity Prices');

% Scatter plot of humidity vs. electricity prices
figure;
scatter(data.Var7, data.Var48);
xlabel('Humidity');
ylabel('Price');
title('Scatter Plot: Humidity vs. Electricity Prices');

% Data Filtering:
% 
% Separates data into training (2015, 2016, 2017) and validation (2018) sets.
% Creates box plots and scatter plots for electricity prices for each year.

% Filter data for training (2015, 2016, 2017) and validation (2018)
train_data = data(year(data.Var1) >= 2015 & year(data.Var1) <= 2017, :);
validation_data = data(year(data.Var1) == 2018, :);

% Box plot of electricity prices for each year
figure;
boxplot(data.Var48, year(data.Var1));
xlabel('Year');
ylabel('Price');
title('Box Plot: Electricity Prices for Each Year');

% Extract unique years from the dataset
unique_years = unique(year(data.Var1));

% Create scatter plots for each year
for i = 1:length(unique_years)
    year_data = data(year(data.Var1) == unique_years(i), :);
    
    figure;
    scatter(year_data.Var1, year_data.Var48);
    xlabel('Timestamp');
    ylabel('Price');
    title(['Scatter Plot: Electricity Prices for Year ' num2str(unique_years(i))]);
end

% Define selected columns
%Var48 is the price actual
selected_columns = {'Var4', 'Var7', 'Var8', 'Var9', 'Var10','Var11', 'Var14','Var23', 'Var24', 'Var25','Var26','Var27','Var28', 'Var32','Var33','Var34', 'Var37', 'Var39', 'Var41', 'Var42', 'Var44', 'Var45', 'Var46', 'Var47','Var48'};

% Select columns for training data
selected_train_data = train_data(:, selected_columns);

% Remove rows with missing values from training data
selected_train_data = rmmissing(selected_train_data);

% Separate predictors and response variable for training data
X_train = table2array(selected_train_data(:, 1:end-1));  % Predictors
y_train = table2array(selected_train_data(:, end));      % Response variable (electricity prices)

% Fit linear regression model on training data
mdl = fitlm(X_train, y_train);

% Select columns for validation data
selected_validation_data = validation_data(:, selected_columns);

% Remove rows with missing values from validation data
selected_validation_data = rmmissing(selected_validation_data);

% Separate predictors and response variable for validation data
X_validation = table2array(selected_validation_data(:, 1:end-1));  % Predictors
y_validation = table2array(selected_validation_data(:, end));      % Response variable (electricity prices)

% Predict using the trained model on validation data
y_pred = predict(mdl, X_validation);

% Display the regression results
disp(mdl);
% Plotting the trained model
figure;
plotResiduals(mdl, 'fitted');
title('Residuals Plot');


% Calculate Root Mean Squared Error (RMSE)
rmse = sqrt(mean((y_pred - y_validation).^2));
% Plotting y_pred vs. y_validation
figure;
scatter(y_validation, y_pred);
hold on;
plot([min(y_validation), max(y_validation)], [min(y_validation), max(y_validation)], 'k--', 'LineWidth', 2);
xlabel('Actual Prices');
ylabel('Predicted Prices');
title('Actual vs. Predicted Prices');
legend('Predictions', 'Ideal Line');


% Display RMSE value
disp(['Root Mean Squared Error (RMSE): ', num2str(rmse)]);

% Plotting the maximum and minimum errors
errors = y_pred - y_validation;
figure;
subplot(2, 1, 1);
plot( errors, 'bo-');
xlabel('Timestamp');
ylabel('Errors');
title('Model Errors Over 2018 Data');

subplot(2, 1, 2);
histogram(errors, 20);
xlabel('Errors');
ylabel('Frequency');
title('Histogram of Model Errors');

% Generating random points for prediction
rng(42); 
num_random_points = 5;
random_indices = randi(size(X_validation, 1), num_random_points, 1);
random_points_X = X_validation(random_indices, :);
random_points_y_actual = y_validation(random_indices);

% Predicting on random points
random_points_y_pred = predict(mdl, random_points_X);

% Plotting random points predictions
figure;
bar(random_points_y_pred);
xlabel('Random Points');
ylabel('Predicted Prices');
title('Random Points: Predicted Prices');

% Define selected columns
selected_columns = {'Var4', 'Var7', 'Var8', 'Var9', 'Var10', 'Var11', 'Var14', 'Var23', 'Var24', 'Var25', 'Var26', 'Var27', 'Var28', 'Var32', 'Var33', 'Var34', 'Var37', 'Var39', 'Var41', 'Var42', 'Var44', 'Var45', 'Var46', 'Var47', 'Var48'};

% Select columns for the original data
selected_data = data(:, selected_columns);

% Specify the folder path where you want to save the file
folderPath = 'C:\Users\Shourya\Documents\MATLAB\unit 3 matlab\matlab';

% Save the selected columns data to a CSV file in the specified folder
writetable(selected_data, fullfile(folderPath, 'selected_data.csv'));

