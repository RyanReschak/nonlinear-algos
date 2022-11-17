%Ryan Reschak, Scott Crowner
clear
clc

syms x1 x2 alpha

%x_0 = starting point
%func = Symbolic expression made of your vars
%vars = a vector of your symbolic variables ex: [x_1, x_2]
%k = max iterations before termination

%Equation 1:
%x_0 = [5, 20];
%vars = [x1, x2];
%func = 2*x1*x2 + 3*x2 - x1^2 -2*x2^2;
%Equation 2:
%x_0 = [5, 20];
vars = [x1,x2];
func = -abs(x1)-abs(x2)-2*sin(x1)-3*sin(x2)+5; 
%func = -abs(x1)-sin(x1);
%func = -abs(x2)-2*sin(x1)-1.5*sin(x2);
%Equation 3
%vars = [x1, x2];
%func = -(2*x1-1)*(x1+1)*(x1-2)*(x1+2)-(x2-2)*(x2+2);

k = 10;

%% Multiple Start Point Steepest Ascent
x_set = randi([-1000,1000], 5, 2);
grad = gradient(func,vars);
i = 0;
tic
for j = 1:length(x_set(:,1))
    x_final = x_set(j,:);
    x_prev = x_final+100;
    while abs(norm(x_final-x_prev)) > 0.001 & i < k
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(simplify(func),vars,alpha_eq')))==0, alpha, 'PrincipalValue', true); 
        
        x_prev = x_final;
        x = x_final' + alpha_opt*subs(grad, vars, x_final);
        x_final = double(x');
        i = i + 1;
    end
    if j == 1
        x_last_best = x_final;
    end
    if j~=1 & double(subs(func, vars, x_final)) > double(subs(func, vars, x_last_best))
        x_last_best = x_final;
    end
end
toc
double(double(subs(func, vars, x_last_best)))
double(x_last_best)
%% Multiple Start Point Newton's Method

x_set = randi([-1000,1000], 5, 2);
grad = gradient(func,vars);
H = hessian(func, vars);
i = 0;
tic
for j = 1:length(x_set(:,1))
    x_final = x_set(j,:);
    x_prev = x_final+100;
    while abs(norm(x_final-x_prev)) > 0.001 & i < k
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(func,vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);
        
        x_prev = x_final;
        x = x_final' + alpha_opt*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        x_final = double(x');
        i = i + 1;
    end
    if j == 1
        x_last_best = x_final;
    end
    if j~=1 & double(subs(func, vars, x_final)) > double(subs(func, vars, x_last_best))
        x_last_best = x_final;
    end
end
toc
double(double(subs(func, vars, x_last_best)))
double(x_last_best)