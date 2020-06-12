function Adj_mat = rgg(num_vertices)
%% Function to generate a random graph and return the adjacency matrix
%   Input:  num_vertices = num of vertices
%  Output:       Adj_mat = n * n matrix such that (Adj_mat)_{ij} = edge_weight_{ij}
%%
      if nargin ~= 1
          error('Incorrect nu. of inputs: usage Adj_mat = rgg(num_of_vertices)');
      end
      
      if (num_vertices  <= 0)
         error('No. of vertices must be positive')
      end
      
%%
      lo = 1; hi = 10;
      Weight  = lo + (hi-lo+1) *rand(num_vertices);
      Weight = 0.5 .* (Weight + Weight');
      
      Prob_mat = rand(num_vertices);
      Tmp      = Prob_mat >= 0.5;  
      Tmp = triu(Tmp,1);  Tmp = Tmp + Tmp';
      Adj_mat  = Tmp .* Weight;
    
end