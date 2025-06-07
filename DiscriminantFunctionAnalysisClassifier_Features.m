% Discriminant Function Analysis classifier
% using the extracted features of neurons


function predictedGroup = discriminant_classify(trainData, testData)
% train the DFA classifier using the training data and evaluate grades for
% the test data.
%
%   Inputs: 
%       trainData: training data.
%       testData: test data.
%   
%   Outputs:
%       predictedGroup: predicted grades.
%       posteriorProbabilities: probability of prediction.

    % Train the Discriminant Analysis Classifier
    discriminantModel = fitcdiscr(trainData(:,2:end), trainData(:,1), ...
        'discrimType', 'diaglinear');
    
    % Predict Group Membership and Probabilities for test data
    [predictedGroup, ~] = ...
        predict(discriminantModel, testData(:,2:end));
end


function [trainData, testData] = shuffle_divide(data)
% shuffle data and take equal number from each grade as training data,
% and the rest is test data.
%
%   Inputs: 
%       data: all data.
%   
%   Outputs:
%       trainData: training data.
%       testData: test data.

    allLen = height(data);
    allIdx = randperm(allLen); % shuffle
    data = data(allIdx, :);

    grade1 = data(data(:,1)==1,:);
    grade2 = data(data(:,1)==2,:);
    
    rowNum = 37; % for the train data
    trainData = [grade1(1:rowNum,:); grade2(1:rowNum,:)]; % divide
    testData = [grade1(rowNum+1:end,:); grade2(rowNum+1:end,:)];
end


function [trainData, testData] = shuffle_grade(data)
% shuffle data with its grade types randomised,
% and then take equal number from each grade to form training data,
% leaving the rest as test data.
%
%   Inputs: 
%       data: all data.
%   
%   Outputs:
%       trainData: training data.
%       testData: test data.

    allLen = height(data);
    allIdx = randperm(allLen); % shuffle

    % Shuffle grade
    grade = data(:,1);
    grade = grade(allIdx);
    data(:,1) = grade;

    grade1 = data(data(:,1)==1,:);
    grade2 = data(data(:,1)==2,:);
    
    rowNum = 37; % for the train data
    trainData = [grade1(1:rowNum,:); grade2(1:rowNum,:)]; % divide
    testData = [grade1(rowNum+1:end,:); grade2(rowNum+1:end,:)];
end


%% Main script

performance = zeros(100,2);

% Test the original data 100 times randomly shuffled.
for i = 1:100
    [trainData, testData] = shuffle_divide(data);

    predicted_group = discriminant_classify(trainData, testData);
    
    % Calculate performance
    prediction = predicted_group == testData(:, 1);
    performance(i,1) = sum(prediction)/height(prediction);
end

% Test the data with randomised grade types 100 times randomly shuffled.
for i = 1:100
    [trainData, testData] = shuffle_grade(data);

    predicted_group = discriminant_classify(trainData, testData);
    
    % Calculate performance
    prediction = predicted_group == testData(:, 1);
    performance(i,2) = sum(prediction)/height(prediction);
end