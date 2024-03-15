# def fibonacci_mai_mici(n):
#     a, b = 0, 1
#     while a < n:
#         print(a, end=' ')
#         a, b = b, a + b
#
# fibonacci_mai_mici(5)
#
#
# print('\n')
# def este_prim(n):
#     if n <= 1:
#         return False
#     elif n <= 3:
#         return True
#     elif n % 2 == 0 or n % 3 == 0:
#         return False
#     i = 5
#     while i * i <= n:
#         if n % i == 0 or n % (i + 2) == 0:
#             return False
#         i += 6
#     return True
#
#
# numar=5
# if este_prim(numar):
#     print(numar, "este un număr prim.")
# else:
#     print(numar, "nu este un număr prim.")
#


def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return

    numar_corecte = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    acuratete = numar_corecte / len(y_true)
    return acuratete


y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

#print(accuracy_score(y_true, y_pred))


def precision_recall_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return

    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 0)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true == 1)

    precizie = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return round(precizie, 2), round(recall, 2)

# print(precision_recall_score(y_true, y_pred))

def mse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return

    mse = sum((pred-true)**2 for true, pred in zip(y_true, y_pred)) / len(y_true)
    return mse

print(mse(y_true, y_pred))






