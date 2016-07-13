require 'torch'   -- torch
require 'optim'
require 'rnn'      -- provides a normalization operator

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

local f_file = io.open(arg[1], 'r')
local p_file = io.open(arg[2], 'w')
local i = 0
for line in f_file:lines('*l') do
	i = i + 1
end
local data = torch.Tensor(i, 351)
i = 0
local names = {}
local line_counter = 0
local f_file = io.open(arg[1], 'r')
for line in f_file:lines('*l') do
	i = i+1
	line_counter = line_counter+1
	local l = line:split(',')
	first = true
	for key, val in ipairs(l) do
	  if first == false then
	  data[i][key] = val
	  else data[i][key] = line_counter
	  names[i] = val
	  first = false
	  end
	end
end
local X = data[{{},{2,-1}}]
model = torch.load('tracking_model.dat')
local myPrediction = model:forward(X)
p_file:write('NAME,F1,F2,F3,F4\n')
for p=1, (#myPrediction)[1] do
	p_file:write(names[p]..','..tostring(1000*myPrediction[p][1])..','..tostring(1000*myPrediction[p][2])..','..tostring(1000*myPrediction[p][3])..','..tostring(1000*myPrediction[p][4])..'\n')

end
