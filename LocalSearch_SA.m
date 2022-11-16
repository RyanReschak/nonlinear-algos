%Ryan Reschak
syms x1 x2 alpha

%x_0 = starting point
%func = Symbolic expression made of your vars
%vars = a vector of your symbolic variables ex: [x_1, x_2]
%k = max iterations before termination
x_0 = [0.5, 0.5];
x_prev = x_0+100;
vars = [x1, x2];
k = 10;
func = 2*x1*x2 + 3*x2 - x1^2 -2*x2^2;
t = 0.5;
annel = 0.1;
annel_iters = 2;
%% Steepest Ascent
x_final = x_0;
grad = gradient(func,vars)
i = 0;
tic
while abs(norm(x_final-x_prev)) > 0.001
    x_prev = x_final;
    y_old = subs(func, vars, x_final);
    for j = 1:annel_iters
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(grad, vars, x_final);
        alpha_opt = solve(diff(subs(func,vars,alpha_eq'))==0, alpha); 
        
        x = x_final' + alpha_opt*subs(grad, vars, x_final);
        
        %cost calculation
        cost = y_old - subs(func, vars, x');
        if cost <= 0
            x_final = x';
        else
            
            p = rand;
            if p  < exp(-cost/t)
                x_final = x';
            else
                %do the wrong thing on purpose
                x = x_final' - alpha_opt*subs(grad, vars, x_final)
                x_final = x';
            end
        end
        y_old = subs(func, vars, x');
        i = i + 1;
    end
    t = t - annel; 
end
toc
%% Newtons method
x_final = x_0;
grad = gradient(func,vars)
H = hessian(func, vars)
i = 0;
tic
while i < k
    %find alpha equation
    alpha_eq = x_final' + alpha*inv(H)*subs(grad, vars, x_final)
    alpha_opt = solve(diff(subs(func,vars,alpha_eq'))==0, alpha) 
    
    x = x_final' + alpha_opt*inv(H)*subs(grad, vars, x_final)
    x_final = x';
    i = i + 1;
end
toc
