% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
% A finite difference solution to the 2D Laplacian and Poisson equation
% u_xx + u_yy = f(x,y) with iterative solution techniques
% with an exact solution of u_e = (x-x^2)(y-y^2) and homogeneous boundary
% conditions.
% 
% ME310: Numerical Methods Spring 25-26
% Author: Atakan Aygun
%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

clear all
close all
clc

% Create Domain

L = 1;         % Domain length
N = 10;        % Number of points in one direction
N2 = N*N;      % Total number of points
h = L/(N+1);   % Spacing between poitns (Uniform spacing)

x = linspace(0,1,N+2);
y = linspace(0,1,N+2);
[X,Y] = meshgrid(x(2:end-1),y(2:end-1));
x = reshape(X,N2,1);
y = reshape(Y,N2,1);

% Initialize the coefficient array A
A = zeros(N2,N2);

% Fill the coefficient array for the second order central differencing
for i = 1:N2
    idE = i+1;
    idW = i-1;
    idN = i+N;
    idS = i-N;
    if i == 1 
        % Bottom left corner
        A(i,i)   = 4;
        A(i,idE) = -1;
        A(i,idN) = -1;
    elseif i == N
        % Bottom right corner
        A(i,i)   = 4;
        A(i,idN) = -1;
        A(i,idW) = -1;
    elseif i == N2-N+1
        % Top left corner
        A(i,i)   = 4;
        A(i,idE) = -1;
        A(i,idS) = -1;
    elseif i == N2
        % Top right corner
        A(i,i)   = 4;
        A(i,idW) = -1;
        A(i,idS) = -1;
    elseif i>1 && i < N
        % Bottom boundary y = 0
        A(i,i) = 4;
        A(i,idE) = -1;
        A(i,idN) = -1;
        A(i,idW) = -1;
    elseif i>(N2-N+1) && i<N2
        % Top Boundary y = 1
        A(i,i)   = 4;
        A(i,idE) = -1;
        A(i,idW) = -1;
        A(i,idS) = -1;
    elseif mod(i,N) == 1 && (i>1 || i<(N2-Nx+1))
        % Left Boundary x = 0
        A(i,i)   = 4;
        A(i,idE) = -1;
        A(i,idN) = -1;
        A(i,idS) = -1;
      
    elseif mod(i,N)==0 && (i>N||i<N2)
        % Right boundary x = 1
        A(i,i)   = 4;
        A(i,idN) = -1;
        A(i,idW) = -1;
        A(i,idS) = -1;
    else
        A(i,i) = 4;
        A(i,idE) = -1;  % East Neighbor
        A(i,idN) = -1;  % North Neighbor      
        A(i,idW) = -1;  % West Neighbor
        A(i,idS) = -1;  % South Neighbor
    end

end

% Calculate the exact solution u_e
global u_e
u_e = (x - x.^2) .* (y - y.^2);

% Calculate the rhs vector for this exact solution
b = A*u_e;

u0  = zeros(N2,1);    % Initial guess all zero
TOL = 1e-3;           % Tolerance value to stop iterations

% case 'Jacobi'
[u_Jacobi, resJacobi] = JacobiIteration(A, b, u0, TOL);

% case 'GS'
[u_GS, resGS] = GaussSeidel(A, b, u0, TOL);

% case 'SOR'
w = 2 / (1 +sin(pi*h));
[u_SOR, resSOR] = SOR(A, b, u0, w, TOL);

% case 'Default'
timeStart = tic;
u_matlab = A\b;
timeStop = toc;
fprintf(['Default Matlab backslash solver finishes in %.3f seconds for ' ...
    'N=%d\n'], timeStop-timeStart, N)

% Plot iteration vs error norm plot
fgh = figure; h1 = axes;
semilogy(0:length(resJacobi)-1,resJacobi,'-o')
hold on
semilogy(0:length(resGS)-1,resGS,'-o')
semilogy(0:length(resSOR)-1,resSOR,'-o')
yline(TOL,'r--', LineWidth=1.5);
legend('Jacobi','Gauss-Seidel','SOR', 'Tolerance','Location','NorthEast', Interpreter='latex');
xlabel('Number of Iterations', Interpreter='latex')
ylabel('Error Norm', Interpreter='latex')
title(['N = ',num2str(N)], Interpreter='latex')

fs = 10;
        set(fgh,'Units','centimeters ')
        set(fgh, 'Position', [4.0,4.0,24.0,16.0]);
        set(gca,'FontSize',fs+2)

        % axis square ;
        set(gcf,'Units','inches');
        screenposition = get(gcf,'Position');
        set(gcf,...
            'PaperPosition',[0 0 screenposition(3:4)],...
            'PaperSize',[screenposition(3:4)]);

function [u, resVec] = JacobiIteration(A, b, u, TOL)
% 
% function  : JacobiIteration(A,b,x0,TOL)
% purpose   : Solve Ax=b with Jacobi iterative method with the given
%             tolerance value TOL and initial guess x0
% 
global u_e

N = size(A,1);
resVec = [];

errNorm = Inf;  % Initialize the stopping criteria
itr = 1;
maxitr = 5000;
tstart = tic;
while itr < maxitr
u_old = u;          % Store the previous solution
for i = 1:N
    jSum = 0 ;      % Initialize the sum in the right hand side
    
    if A(i,i) == 0
        fprintf('There is a zero in the diagonal')
        return
    end
    for j = 1:N
        if i~=j
            jSum = jSum + A(i,j)*u_old(j);
        end
    end

    u(i) = (b(i) - jSum) / A(i,i);
end

errNorm = max(abs(u - u_e));
resVec = [resVec, errNorm];
if errNorm < TOL
    break
end
itr = itr + 1;

end

telapsed = toc(tstart);
fprintf(['Jacobi method finishes in %.3f seconds ' ...
    'with %d iterations for N=%d\n'], telapsed, itr, sqrt(N))
end

function [u, resVec] = GaussSeidel(A, b, u, TOL)
% 
% function  : GaussSeidel(A,b,x,TOL)
% purpose   : Solve Ax=b with Gauss-Seidel iterative method with the given
%             tolerance value TOL and initial guess x
% 
global u_e
N = size(A,1);
errNorm = Inf;  % Initialize the stopping criteria
resVec = [];

itr = 1;
maxitr = 5000;
tstart = tic;

while itr < maxitr
uold = u;       % Store the previous level solution 

for i = 1:N
    jSum = 0 ;      % Initialize the sum in the right hand side
    
    if A(i,i) == 0
        % Check if there is a zero on the diagonal
        fprintf('There is a zero in the diagonal')
        return
    end

    for j = 1:i-1
        jSum = jSum + A(i,j)*u(j);
    end

    for j = i+1:N
        jSum = jSum + A(i,j)*uold(j);
    end
    
    u(i) = (b(i) - jSum) / A(i,i);
end

errNorm = max(abs(u - u_e));
resVec = [resVec, errNorm];
if errNorm < TOL
    telapsed = toc(tstart);
    break
end

itr = itr + 1;

end

telapsed = toc(tstart);
fprintf(['Gauss-Seidel method finishes in %.3f seconds ' ...
    'with %d iterations for N=%d\n'], telapsed, itr, sqrt(N))
end

function [u, resVec] = SOR(A, b, u, w, TOL)
% 
% function  : SOR(A,b,x,TOL)
% purpose   : Solve Ax=b with Gauss-Seidel iterative method with the given
%             tolerance value TOL and initial guess x
% 
global u_e
N = size(A,1);
errNorm = Inf;  % Initialize the stopping criteria
resVec = [];

itr = 1;
maxitr = 5000;

tstart = tic;

while itr < maxitr
for i = 1:N
    jSum = 0 ;      % Initialize the sum in the right hand side
    uold = u;       % Store the previous level solution 
    
    if A(i,i) == 0
        % Check if there is a zero on the diagonal
        fprintf('There is a zero in the diagonal')
        return
    end

    for j = 1:i-1
        jSum = jSum + A(i,j)*u(j);
    end

    for j = i+1:N
        jSum = jSum + A(i,j)*uold(j);
    end
    
    u(i) = (1-w)*uold(i) + w*(b(i) - jSum) / A(i,i);
end

errNorm = max(abs(u - u_e));
resVec = [resVec, errNorm];
if errNorm < TOL
    break
end

itr = itr + 1;

end

telapsed = toc(tstart);
fprintf(['SOR method finishes in %.3f seconds ' ...
    'with %d iterations for N=%d with the relaxation' ...
    ' %.2f\n'], telapsed, itr, sqrt(N), w)
end

% function plotSparsity(A)
%     N = size(A,1);
%     N = sqrt(N);
%     fgh =figure(1); h1 = axes;
%     spy(A)
% 
%     saveFigure = 1;
%     if saveFigure == 1
%         title(['Sparsity pattern of A with N = ', num2str(N)])
%         fs = 10;
%         set(fgh,'Units','centimeters ')
%         set(fgh, 'Position', [4.0,4.0,16.0,16.0]);
%         set(gca,'FontSize',fs+2)
% 
%         axis square ;
%         set(gcf,'Units','inches');
%         screenposition = get(gcf,'Position');
%         set(gcf,...
%             'PaperPosition',[0 0 screenposition(3:4)],...
%             'PaperSize',[screenposition(3:4)]);
%         fileName = append('Figures/sparsity_N',num2str(N));
% 
%         print('-dpdf', '-painters', fileName)
%     end
% end
