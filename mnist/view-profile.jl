using Serialization
import PProf
# Modules get serialized, so I need to import them before I deserialize...
import JuMP
import MathProgIncidence
import MathOptAI
allocdata = open("allocdata.bin", "r") do io
    return deserialize(io)
end
PProf.Allocs.pprof(allocdata; webport = 62260)
