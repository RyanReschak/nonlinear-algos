%Ryan Reschak
clear
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
%func = -abs(x1)-abs(x2)-2*sin(x1)-3*sin(x2)+5; 
%func = -abs(x1)-sin(x1);
%func = -abs(x2)-2*sin(x1)-1.5*sin(x2);
%Equation 3
x_0 = [-100, -100];
vars = [x1, x2];
func = -(2*x1-1)*(x1+1)*(x1-2)*(x1+2)-(x2-2)*(x2+2);

k = 20;
annel = 0.1;
annel_iters = 4;
%% Steepest Ascent
x_final = x_0;
grad = gradient(func,vars)
i = 0;
t = 0.9;
x_prev = x_0+100;
tic
while double(norm(x_final-x_prev)) > 0.001 & i < k
    x_prev = double(x_final);
    y_old = double(subs(func, vars, x_final));
    for j = 1:annel_iters
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(simplify(func),vars,alpha_eq')))==0, alpha, 'PrincipalValue', true); 
        
        x = double(x_final' + alpha_opt*subs(grad, vars, x_final));
        
        %cost calculation
        cost = double(y_old - subs(func, vars, x'));
        
        p = rand;
        if cost >= 0 | p  < exp(-cost/t)
            x_final = x';
        else
            %do the wrong thing on purpose
            x = x_final' + randi([-5,5], length(vars),1);
            x_final = x';

        end
        
        double(x_final)
        y_old = subs(func, vars, x_final);
        i = i + 1;
    end
    t = t/(t*annel+1); 
end
toc
%% Newtons method
x_final = x_0;
grad = gradient(func,vars);
H = hessian(func, vars);
t = 0.9;
i = 0;
x_prev = x_0+100;
tic
while abs(norm(x_final-x_prev)) > 0.001 & i < k
    x_prev = x_final;
    y_old = subs(func, vars, x_final);
    for j = 1:annel_iters
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        alpha_opt = solve(diff(simplify(subs(func,vars,alpha_eq')))==0, alpha, 'PrincipalValue', true);
        
        if(isempty(alpha_opt))
            x = x_final' + 0.25*subs(inv(H),vars,x_final)*subs(grad, vars, x_final); %if no optimal solution then just alpah = 0.25
        else
            x = x_final' + alpha_opt*subs(inv(H),vars,x_final)*subs(grad, vars, x_final);
        end
        %cost calculation
        cost = double(y_old - subs(func, vars, x'));
        p = rand;
        if cost >= 0 | p  < exp(-cost/t)
            x_final = double(x');
        else
            %do the wrong thing on purpose
            x = x_final' + randi([-3,3], length(vars),1);
            x_final = doulbe(x');

        end
        double(x_final)
        y_old = subs(func, vars, x_final);
        i = i + 1;
    end
    t = t/(t*annel+1); 
end
toc
