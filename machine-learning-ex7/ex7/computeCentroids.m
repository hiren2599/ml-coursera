function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
avg=zeros(K,n);
counter_avg=zeros(K,1);

for i=1:length(idx)
  for j=1:K
    if idx(i)==j
      avg(j,:)=avg(j,:)+X(i,:);
      counter_avg(j)=counter_avg(j)+1;
    endif
  endfor
endfor

for j=1:K
  if counter_avg(j)!=0
    avg(j,:)=avg(j,:)/counter_avg(j);
  endif
endfor

centroids=avg;


% =============================================================


end

