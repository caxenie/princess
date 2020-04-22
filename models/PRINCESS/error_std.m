% PRINCESS: Prediction of Individual Breast Cancer Evolution to Surgical Size
% Error std dev calculation
% above a given threshold y, the measurement error is sub-proportional and, 
% below this threshold, the error made is the same as when measuring y.
% as from [Benzekry et al., 2014c]
function s = error_std(alfa, sigma, M, y)
N = length(M);
s = zeros(1, N);
for id=1:N
    if(M(id) >= y(id))
        s(id) = sigma * M(id)^alfa;
    else
        s(id) = sigma * y(id)^alfa;
    end
end
end