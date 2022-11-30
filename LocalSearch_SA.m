%% Simulated annealing global optimum search method tester
% This program employs simulated annealing to test its capability of
% finding the global maximum among three example functions.
% Usage: Run this first section first, then run the desired test section
% Made by: Ryan Reschak, Scott Crowner
clear
clc

% Data cell array for exporting
Data = cell(2,6);
Data{1,2} = 'avg time'; Data{1,3} = 'max time'; Data{1,4} = 'min time'; Data{1,5} = 'success rate'; Data{1,6} = 'best soln';

syms x1 x2 alpha

%x_0 = starting point
%func = Symbolic expression made of your vars
%vars = a vector of your symbolic variables ex: [x_1, x_2]
%k = max iterations before termination

n = 2; % CHANGES FUNCTION TO BE USED

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
        actual = 7.28;
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
annel = 0.1;
annel_iters = 4;

%% SA Random Search with Newtons Method
% Chooses a random starting point on the domain x1, x2 = [-100, 100].
% Runs this test 5 times

trials = 5; % Number of trials
best_value_list = zeros(1,trials);   % List of results for each of the five test trials
f_best_truth = zeros(1,trials);      % Array of booleans tracking how often the method found the correct solution
endTime = zeros(1,trials);           % List of run durations

% Loop through trials
for attempt = 1:trials
    
    %Three major parameters: starting x, k, and NN. 
    i = 0;
    t = 0.9;
    x_final = randi([-100,100], 1, length(vars));
    NN = [-5,5];
    k = 20;

    startTime = tic;    % START TIMER
    
    % Simulated Annealing
    while i < k
        
        y_old = double(subs(func, vars, x_final));
        for j = 1:annel_iters
            
            %Check Nearest Neighbors
            x = double(x_final' + randi(NN, length(vars),1));
            
            %cost calculation
            cost = double(y_old - subs(func, vars, x'));
            
            %Acceptance if cost is negative or based on the probability
            %cost is negative if y_old is less than x which we are trying to
            %maximize (opposite sign if minimize).
            p = rand;
            if cost <= 0 | p  < exp(-cost/t)
                x_final = x';
                y_old = subs(func, vars, x_final);
            end
            
        end
        i = i + 1;
        t = t/(t*annel+1); 
    end
    double(x_final)
    double(subs(func, vars, x_final))
    % toc
    
    %With Newtons Method Improvement
    %The hope is that with SA it gets it in the ball park and then NM
    %it gets closer
    x_better = x_final;
    grad = gradient(func,vars);
    H = hessian(func, vars);
    x_prev = x_better + 100;
    % tic
    while abs(norm(x_better-x_prev)) > 0.001
        x_prev = x_better;
        
        %find alpha equation
        alpha_eq = x_better' + alpha*subs(inv(H),vars,x_better)*subs(grad, vars, x_better);
        alpha_opt = solve(diff(simplify(subs(func,vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);
        
        if(isempty(alpha_opt))
            x = x_better' + 0.25*subs(inv(H),vars,x_better)*subs(grad, vars, x_better); %if no optimal solution then just alpha = 0.25
        else
            x = x_better' + alpha_opt*subs(inv(H),vars,x_better)*subs(grad, vars, x_better);
        end
        x_better = double(x');
    end
    double(x_better);   % Get x for the best solution
    best_value = double(subs(func, vars, x_better));    % Get f(x) for the best solution
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
%% SA Random Search with Steepest Ascent Method
% Chooses a random starting point on the domain x1, x2 = [-100, 100].
% Runs this test 5 times
[s1,s2] = RandStream.create('mrg32k3a', 'NumStreams', 2, 'Seed','shuffle')

%Three major parameters: starting x, k, and NN. 
i = 0;
t = 0.9;

x_final = randi(s1, [-100,100], 1, length(vars));
NN = [-5,5];
k = 20;

trials = 5; % Number of trials
best_value_list = zeros(1,trials);   % List of results for each of the five test trials
f_best_truth = zeros(1,trials);      % Array of booleans tracking how often the method found the correct solution
endTime = zeros(1,trials);           % List of run durations

% Loop through trials
for attempt = 1:trials

startTime = tic;    % START TIMER

% Simulated Annealing
while i < k
    
    y_old = double(subs(func, vars, x_final));
    for j = 1:annel_iters
        
        %Check Nearest Neighbors
        x = double(x_final' + randi(s2, NN, length(vars),1));
        
        %cost calculation
        cost = double(y_old - subs(func, vars, x'));
        
        %Acceptance if cost is negative or based on the probability
        %cost is negative if y_old is less than x which we are trying to
        %maximize (opposite sign if minimize).
        p = rand;
        if cost <= 0 | p  < exp(-cost/t)
            x_final = x';
            y_old = subs(func, vars, x_final);
        end
        
    end
    i = i + 1;
    t = t/(t*annel+1); 
end
double(x_final)
double(subs(func, vars, x_final))
% toc

%With Steepest Ascent Method Improvement
%The hope is that with SA it gets it in the ball park and then NM
%it gets closer
x_better = x_final;
grad = gradient(func,vars);
x_prev = x_better + 100;

while abs(norm(x_better-x_prev)) > 0.001
    x_prev = x_better;
    
    %find alpha equation
    alpha_eq = x_better' + alpha*subs(grad, vars, x_better);
    alpha_opt = solve(diff(simplify(subs(func,vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);
    
    %if(isempty(alpha_opt))
        %x = x_better' + 0.25*subs(inv(H),vars,x_better)*subs(grad, vars, x_better); %if no optimal solution then just alpha = 0.25
    %else
    x = x_better' + alpha_opt*subs(grad, vars, x_better);
    x_better = double(x');
end
double(x_better);   % Get x for the best solution
best_value = double(subs(func, vars, x_better));    % Get f(x) for the best solution
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
