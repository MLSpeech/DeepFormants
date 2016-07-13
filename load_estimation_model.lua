require 'torch'   -- torch
require 'optim'
require 'nn'      -- provides a normalization operator

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

local f_file = io.open(arg[1], 'r')
local p_file = io.open(arg[2], 'w')
local data = torch.Tensor(1, 351)
local name = ''
for line in f_file:lines('*l') do
local l = line:split(',')
first = true
	for key, val in ipairs(l) do
	  if first == false then
	  data[1][key] = val
	  else data[1][key] = 0
	  first = false
	  name = val
	  end
	end
end

local X = data[{{},{2,-1}}]
model = torch.load('estimation_model.dat')
local myPrediction = model:forward(X)
p_file:write('NAME,F1,F2,F3,F4\n')
p_file:write(name..','..tostring(1000*myPrediction[1][1])..','..tostring(1000*myPrediction[1][2])..','..tostring(1000*myPrediction[1][3])..','..tostring(1000*myPrediction[1][4])..'\n')

