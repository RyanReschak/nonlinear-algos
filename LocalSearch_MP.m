%Ryan Reschak, Scott Crowner
clear
clc

syms x1 x2 alpha

%x_0 = starting point
%func = Symbolic expression made of your vars
%vars = a vector of your symbolic variables ex: [x_1, x_2]
%k = max iterations before termination

%Equation 1:
% vars = [x1, x2];
% func = 2*x1*x2 + 3*x2 - x1^2 -2*x2^2;

%Equation 2:
% vars = [x1,x2];
% func = -abs(x1)-abs(x2)-2*sin(x1)-3*sin(x2)+5; 
% func = -abs(x1)-sin(x1);
% func = -abs(x2)-2*sin(x1)-1.5*sin(x2);

%Equation 3
vars = [x1, x2];
func = -(2*x1-1)*(x1+1)*(x1-2)*(x1+2)-(x2-2)*(x2+2);

figure(1)
fsurf(func,[-2 2 -2 2])
figure(2)
fcontour(func,[-10 10 -10 10])

k = 3;

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

tic
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
[best_value, best_index] = max(f_final_list)
x_best = x_final_list(best_index, :)
toc

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

tic
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
[best_value, best_index] = max(f_final_list)
x_best = x_final_list(best_index, :)
toc
