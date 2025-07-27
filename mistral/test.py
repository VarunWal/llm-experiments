# def decorator(func):
#     def wrapper():
#         print("Before function call")
#         func()
#         print("After function call")
#     return wrapper
 
# @decorator
# def say_hello():
#     print("Hello!")
 
# say_hello()

def func(a, b=()):
    b.append(a)
    return b

print(func(1))
print(func(2))