x = 10

x + 1

x = 1 + 1

x = "Hello World"

x = 1.0

y = -3

X = "My String"

customary_phrase = "Hello World!"

UniversalDeclarationOfHUmanRightsStart = "人人生而自由，在尊严和权利上一律平等"

# Unicode names (in UTF-8 encoding) are allowed:

δ = 0.00001

안녕하세요 = "Hello"

pi = 3

println(pi)

sqrt = 4


sqrt(100)

# Assignment expressions and assignment versus mutation

a = (b = 2 + 2) + 3

a

b


a = [1 ,2 ,3]

b = a

a[1] = 42

a = 3.14159

b

typeof(1)

Sys.WORD_SIZE

0x1

typeof(0x1)


typemin(Int32)

typemax(Int32)

for T in [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
    println("$(lpad(T, 7)): [$(typemin(T)), $(typemax(T))]")
end

lpad("Hello", 10)
lpad(10,2)

0.0 == -0.0

bitstring(0.0)

bitstring(-0.0)

1/Inf

1/0

-5 / 0

eps(Float32)

eps(Float64)

eps()

X = typemin(Int64)

X = X - 1

y = BigInt(typemin(Int64))

y = y - 1


setrounding(BigFloat, RoundUp) do
    BigFloat(1) + parse(BigFloat, "0.1")
    
end

setrounding(BigFloat, RoundDown) do
    BigFloat(1) + parse(BigFloat, "0.1")
end

setprecision(40) do 
    BigFloat(1) + parse(BigFloat, "0.1")
end

x = 3

2x^2 - 3x + 1

1.5x^2 - .5x + 1

2^2x

(x-1)x

(x-1)(x+1)

x(x+1)


1 -1
1 -2

3 * 2/12

NaN * false

false * NaN

~123

123 & 234

123 | 234

123 ⊻ 234

xor(123, 234)

nand(123, 234)

123 ⊼ 234


[1,2,3] .^ 3

[1 NaN]

[1 NaN] == [1 NaN]

isequal(NaN,NaN)

isequal([1 NaN], [1 NaN])

isequal(NaN,NaN32)

0.0 == -0.0

isequal(0.0, -0.0)

1 < 2 <= 2 < 3 == 3 > 2 >= 1 == 1 < 3 != 5


v(x) = (println(x); x)

v(1) < v(2) <= v(3)

v(1) > v(2) <= v(3)

1 + 2im

1 + 2im == 1 + 2im

(1+2im) * (1-2im)

(1 + 2im) * (2 - 3im)


(-1 + 2im)^2.5

conj(1 + 2im)

angle(1 + 2im)

sqrt(-1)

sqrt(-1 + 0im)

2//3

6//9

numerator(2//3)

denominator(6//9)


str = "Hello, world.\n"

str[1]

Int(str[1])


str[1:5]

str[begin]


str[end]


v1 = [1,2,3]

"v: $v1"


println("I have $(length(v1)) elements for you.")

println("I have \$100 in my account")

findfirst('z',"xylophone")

findnext('o',"xylophone",1)

occursin("xylophone","The xylophone is a musical instrument")


join(["Apples", "Bananas", "Pineapples"], ", ", " and ")

split("Apples, Bananas, Pineapples", ", ")

function f(x,y) 
    return x + y
end

f(10,20)


f(x,y) = x + y


f(3,4)

g = f

g(3,4)

f(23,56)

∑(x,y) = x + y

∑(3,4)
