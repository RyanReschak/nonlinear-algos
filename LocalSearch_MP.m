%Ryan Reschak, Scott Crowner
clear
clc

Data = cell(2,6);
Data{1,2} = 'avg time'; Data{1,3} = 'max time'; Data{1,4} = 'min time'; Data{1,5} = 'success rate'; Data{1,6} = 'best soln';

syms x1 x2 alpha

%x_0 = starting point
%func = Symbolic expression made of your vars
%vars = a vector of your symbolic variables ex: [x_1, x_2]
%k = max iterations before termination

n = 1;

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

figure(1)
fsurf(func,[-10 10 -10 10])
figure(2)
fcontour(func,[-10 10 -10 10])

k = 10;

%% Multiple Start Point Steepest Ascent

num_start_points = 5;
x_set = randi([-100,100], num_start_points, 2);
f_initial_list = zeros(num_start_points, 1);
for i=1:length(x_set(:,1))
    f_initial_list(i,:) = subs(func, vars, x_set(i,:));
end
x_final_list = zeros(num_start_points, 2);
f_final_list = zeros(num_start_points, 1);
grad = gradient(func,vars);

best_value_list = zeros(1,5);
f_best_truth = zeros(1,5);
endTime = zeros(1,5);
for attempt = 1:5

startTime = tic;
for j = 1:length(x_set(:,1))
    j
    x_final = x_set(j,:);
    x_prev = x_final+100;
    i = 0;
    while abs(norm(x_final-x_prev)) > 0.001 && i < k
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(simplify(func),vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);

        x_prev = x_final;
        x = x_final' + alpha_opt*subs(grad, vars, x_final);
        x_final = double(x');
        i = i + 1;
    end
    x_final_list(j,:) = x_final;
    f_final_list(j,:) = subs(func, vars, x_final);
end
[best_value, best_index] = max(f_final_list);
x_best = x_final_list(best_index, :);
endTime(attempt) = toc(startTime);
best_value_list(attempt) = best_value;
if abs(best_value-actual) < 0.1
    f_best_truth(attempt) = 1;
end

end
avgTime = mean(endTime);
maxTime = max(endTime);
minTime = min(endTime);
successRate = sum(f_best_truth)/5;
bestValue = max(best_value_list);
Data{2,2} = avgTime; Data{2,3} = maxTime; Data{2,4} = minTime; Data{2,5} = successRate; Data{2,6} = bestValue;
Table = cell2table(Data);
writetable(Table, "data.xlsx",'Sheet',1,'Range','A1');

%% Multiple Start Point Newton's Method
num_start_points = 5;
x_set = randi([-100,100], num_start_points, 2);
f_initial_list = zeros(num_start_points, 1);
for i=1:length(x_set(:,1))
    f_initial_list(i,:) = subs(func, vars, x_set(i,:));
end
x_final_list = zeros(num_start_points, 2);
f_final_list = zeros(num_start_points, 1);
grad = gradient(func,vars);
H = hessian(func, vars);

best_value_list = zeros(1,5);
f_best_truth = zeros(1,5);
endTime = zeros(1,5);
for attempt = 1:5

startTime = tic;
for j = 1:length(x_set(:,1))
    j
    x_final = x_set(j,:);
    x_prev = x_final+100;
    i = 0;
    while abs(norm(x_final-x_prev)) > 0.001 & i < k
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(func,vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);
        
        x_prev = x_final;
        x = x_final' + alpha_opt*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        x_final = double(x');
        i = i + 1;
    end
    x_final_list(j,:) = x_final;
    f_final_list(j,:) = subs(func, vars, x_final);
end
[best_value, best_index] = max(f_final_list);
x_best = x_final_list(best_index, :);
endTime(attempt) = toc(startTime);
best_value_list(attempt) = best_value;
if abs(best_value-actual) < 0.1
    f_best_truth(attempt) = 1;
end

end
avgTime = mean(endTime);
maxTime = max(endTime);
minTime = min(endTime);
successRate = sum(f_best_truth)/5;
bestValue = max(best_value_list);
Data{2,2} = avgTime; Data{2,3} = maxTime; Data{2,4} = minTime; Data{2,5} = successRate; Data{2,6} = bestValue;
Table = cell2table(Data);
writetable(Table, "data.xlsx",'Sheet',1,'Range','A1');

