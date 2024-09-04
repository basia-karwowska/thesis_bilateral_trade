'''
A function to compute profit from the trade defined as:
(p_b-p_s)*1(b>=p_b)*1(s<=p_s)
'''
def profit(buyer_price, seller_price, buyer_valuation, seller_valuation):
    if buyer_valuation >= buyer_price and seller_valuation <= seller_price:
        return buyer_price - seller_price
    return 0