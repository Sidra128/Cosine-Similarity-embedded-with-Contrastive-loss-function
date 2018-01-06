classdef CosSimLoss < dagnn.Loss
  properties
    margin = 1;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      
          outputs{1} = vl_nncosloss(inputs{:}, 'margin', obj.margin);
        
      obj.accumulateAverage(inputs, outputs);
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
          [dzdx1, dzdx2] = vl_nncosloss(inputs{:}, derOutputs{1}, 'margin', obj.margin);
       
      derInputs = {dzdx1, dzdx2, []};
      derParams = {} ;
    end
    
    function obj = CosSimLoss(varargin)
      obj.load(varargin{:}) ;
    end
  end
end

