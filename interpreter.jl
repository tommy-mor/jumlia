abstract type ExprAST end

struct EInt <: ExprAST
    val :: Int
end

struct ENeg <: ExprAST
    expr :: ExprAST
end

struct EAdd <: ExprAST
    left :: ExprAST
    right :: ExprAST
end

interpret(expr::EInt) = expr.val
interpret(expr::ENeg) = - interpret(expr.val)
interpret(expr::EAdd) = interpret(expr.left) + interpret(expr.right)

@test interpret(EAdd(EInt(10), EInt(20))) == 30

