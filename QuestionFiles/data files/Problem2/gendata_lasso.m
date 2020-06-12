function [A,b] = gendata_lasso(m,n,noise,option)
%% function to generate test data for lasso 
%   Input:  m: no. of observations
%           n: no. of features
%       noise: standard deviation 
%      option: 0: no noise
%              1: noise added by gaussian distribution
%              2: noise added as an outlier (selecting any 1 of the
%                 observations)
%%
   if nargin == 0
      m = 500;
      n = 2500;
      noise = 0;
      option = 1;
   elseif nargin ~=4
      disp('usage: usage: [A,b] = gendata_lasso(<no. of observations>, <no. of features>, standard deviation for noise, option)');
      disp('option: 0:no noise , 1:noise added by gaussian distribution, 2: outlier');
      error('incorrect input arguments. ');      
   end
%%
x0 = sprandn(n,1,0.05);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
v = sqrt(0.001)*randn(m,1);
b = A*x0 + v;

  switch option 
      case 0;          
      case 1;
          b = b + noise * rand(size(b));
      case 2;
         b(randi([1,m])) = b(randi([1,m])) + noise * rand;
  end

end