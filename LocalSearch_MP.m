%% Multiple start point global optimum search method tester
% This program employs steepest ascent and Newton's method to test their
% capability of finding the global maximum among three example functions.
% Usage: Run this first section first, then run the desired test section
% Made by: Ryan Reschak, Scott Crowner
clear
clc

% Data cell array for exporting
Data = cell(2,6);
Data{1,2} = 'avg time'; Data{1,3} = 'max time'; Data{1,4} = 'min time'; Data{1,5} = 'success rate'; Data{1,6} = 'best soln';

syms x1 x2 alpha

% x_0 = starting point
% func = Symbolic expression made of your vars
% vars = a vector of your symbolic variables ex: [x_1, x_2]
% k = max iterations before termination

n = 3; % CHANGES FUNCTION TO BE USED

switch n
    case 1
        %Equation 1:
        vars = [x1, x2];
        func = 2*x1*x2 + 3*x2 - x1^2 -2*x2^2;
        actual = 2.25;
        Data{2,1} = 'Equation 1';
    case 2
        %Equation 2:
        vars = [x1,x2];
        func = -abs(x1)-abs(x2)-2*sin(x1)-3*sin(x2)+5;
        actual = 7.11;
        Data{2,1} = 'Equation 2';
    otherwise        
        %Equation 3
        vars = [x1, x2];
        func = -(2*x1-1)*(x1+1)*(x1-2)*(x1+2)-(x2-2)*(x2+2);
        actual = 12.8;
        Data{2,1} = 'Equation 3';
end

% Plotting
figure(1)
fsurf(func,[-10 10 -10 10])
figure(2)
fcontour(func,[-10 10 -10 10])

k = 10;

%% Multiple Start Point Steepest Ascent Method
% Uses 5 randomly chosen starting points on the domain x1, x2 = [-100, 100]
% Runs this test 5 times.

% Initialization
num_start_points = 5;   % Number of starting points to test
x_set = randi([-100,100], num_start_points, 2); % Starting x points
f_initial_list = zeros(num_start_points, 1);    % Starting f(x) points
for i=1:length(x_set(:,1))
    f_initial_list(i,:) = subs(func, vars, x_set(i,:));
end
x_final_list = zeros(num_start_points, 2);  % Ending x points
f_final_list = zeros(num_start_points, 1);  % Ending f(x) points
grad = gradient(func,vars); % Gradient

trials = 5; % Number of trials
best_value_list = zeros(1,trials);   % List of results for each of the five test trials
f_best_truth = zeros(1,trials);      % Array of booleans tracking how often the method found the correct solution
endTime = zeros(1,trials);           % List of run durations


% Loop through trials
for attempt = 1:trials

startTime = tic;    % START TIMER
% Loop through each test point
for j = 1:length(x_set(:,1))
    j
    x_final = x_set(j,:); % Initialize point tracker
    x_prev = x_final+100; % Initialize previous loop iteration point - assume 100 is a large enough difference
    i = 0;  % Iteration counter

    % Steepest Acent Method
    while abs(norm(x_final-x_prev)) > 0.001 && i < k
        % Find alpha equation
        alpha_eq = x_final' + alpha*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(simplify(func),vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);
        % Find optimal point
        x_prev = x_final;
        x = x_final' + alpha_opt*subs(grad, vars, x_final);
        x_final = double(x');
        i = i + 1;
    end
    x_final_list(j,:) = x_final;    % Store x final
    f_final_list(j,:) = subs(func, vars, x_final);  % Store function value
end
[best_value, best_index] = max(f_final_list); % Choose f(x) for the best solution
x_best = x_final_list(best_index, :);   % Choose x for the best solution
endTime(attempt) = toc(startTime);  % STOP TIMER
best_value_list(attempt) = best_value;  % Store solution for this trial
if abs(best_value-actual) < 0.1
    f_best_truth(attempt) = 1;  % Store whether the correct solution was found
end

end

% Select data to export:
avgTime = mean(endTime);
maxTime = max(endTime);
minTime = min(endTime);
successRate = sum(f_best_truth)/5;
bestValue = max(best_value_list);
Data{2,2} = avgTime; Data{2,3} = maxTime; Data{2,4} = minTime; Data{2,5} = successRate; Data{2,6} = bestValue;
Table = cell2table(Data);
writetable(Table, "data.xlsx",'Sheet',1,'Range','A1');

%% Multiple Start Point Newton's Method
% Uses 5 randomly chosen starting points on the domain x1, x2 = [-100, 100]
% Runs this test 5 times (trials).

% Initialization
num_start_points = 5;   % Number of starting points to test
x_set = randi([-100,100], num_start_points, 2); % Starting x points
f_initial_list = zeros(num_start_points, 1);    % Starting f(x) points
for i=1:length(x_set(:,1))
    f_initial_list(i,:) = subs(func, vars, x_set(i,:));
end
x_final_list = zeros(num_start_points, 2);  % Ending x points
f_final_list = zeros(num_start_points, 1);  % Ending f(x) points
grad = gradient(func,vars); % Gradient
H = hessian(func, vars);    % Hessian

trials = 5; % Number of trials
best_value_list = zeros(1,trials);   % List of results for each of the five test trials
f_best_truth = zeros(1,trials);      % Array of booleans tracking how often the method found the correct solution
endTime = zeros(1,trials);           % List of run durations

% Loop through trials
for attempt = 1:trials

startTime = tic;    % START TIMER
% Loop through each test point
for j = 1:length(x_set(:,1))
    j
    x_final = x_set(j,:); % Initialize point tracker
    x_prev = x_final+100; % Initialize previous loop iteration point - assume 100 is a large enough difference
    i = 0;  % Iteration counter

    % Newton's Method
    while abs(norm(x_final-x_prev)) > 0.001 & i < k
        % Find alpha equation
        alpha_eq = x_final' + alpha*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(func,vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);
        % Find optimal point
        x_prev = x_final;
        x = x_final' + alpha_opt*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        x_final = double(x');
        i = i + 1;
    end
    x_final_list(j,:) = x_final;    % Store x final
    f_final_list(j,:) = subs(func, vars, x_final); % Store function value
end
[best_value, best_index] = max(f_final_list); % Choose f(x) for the best solution
x_best = x_final_list(best_index, :);   % Choose x for the best solution
endTime(attempt) = toc(startTime);  % STOP TIMER
best_value_list(attempt) = best_value;  % Store solution for this trial
if abs(best_value-actual) < 0.1
    f_best_truth(attempt) = 1;  % Store whether the correct solution was found
end

end

% Select data to export:
avgTime = mean(endTime);
maxTime = max(endTime);
minTime = min(endTime);
successRate = sum(f_best_truth)/5;
bestValue = max(best_value_list);
Data{2,2} = avgTime; Data{2,3} = maxTime; Data{2,4} = minTime; Data{2,5} = successRate; Data{2,6} = bestValue;
Table = cell2table(Data);
writetable(Table, "data.xlsx",'Sheet',1,'Range','A1');

