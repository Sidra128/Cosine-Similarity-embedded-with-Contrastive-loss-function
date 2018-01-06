function [ y1, y2 ] = vl_nncosloss( x1, x2, c, dzdy, varargin )
%%
% For documentation and formulas, please refer to the link below:
%
% https://www.slideshare.net/CenkBircanolu/a-comparison-of-loss-function-on-deep-embedding
% 

%%
if nargin > 3 && ischar(dzdy), varargin = [dzdy, varargin]; end
opts = vl_argparse(opts, varargin);

sx1 = size(x1); sx2 = size(x2); nel = size(x1, 4);
assert(numel(sx1) == numel(sx2), 'Invalid dimensionality');
assert(all(sx1 == sx2), 'Invalid input sizes.');
assert(numel(c) == nel, 'Invalid number of labels.');

x1 = reshape(x1, [], nel);
x2 = reshape(x2, [], nel);
margin=0.2;
cosSim = zeros(nel);
for b =1: size(x1,2)
    cosSim(b) = (x1(:,b)'*x2(:,b))/sqrt(((x1(:,b)'*x1(:,b))*(x2(:,b)'*x2(:,b))));
end
if nargin < 4 || isempty(dzdy) || ischar(dzdy)
    
    loss= cosSim;
    loss(c==1)= 1-cosSim(c==1);
    loss(c==-1)= max(0,cosSim(c==-1) - margin);
    LossAccum = sum(loss); y1 =LossAccum;
else
    
    for b =1: size(x1,2)
        dcosSim_dx1(:,b) = -(x2(:,b)/(sqrt((x1(:,b)'*x1(:,b))*(x2(:,b)'*x2(:,b)))))+...
            ((cosSim(b))* x1(:,b)/(x1(:,b)'*x1(:,b)));
        if c(b)==-1
            dcosSim_dx1(:,b) = - dcosSim_dx1(:,b);
        end
        dcosSim_dx2(:,b) = -(x1(:,b)/(sqrt((x1(:,b)'*x1(:,b))*(x2(:,b)'*x2(:,b)))))+...
            ((cosSim(b))* x2(:,b)/(x2(:,b)'*x2(:,b)));
        if c(b)==-1
            dcosSim_dx2(:,b) = - dcosSim_dx2(:,b);
        end
      
    end
    
    y1 = reshape(dcosSim_dx1, sx1);
    y2 = reshape(dcosSim_dx2, sx2);
end

end
