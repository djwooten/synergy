import re

def get_preceeding_factor(s, idx, close_to_open):
    # return start_idx, substr
    if s[idx-1]==")":
        start_idx = close_to_open[idx-1]
        return start_idx, s[start_idx:idx]
    else:
        # Need to find the closest preceeding +,-,*,/
        prev_operator = 0
        for operator in ["+", "-", "*", "/"]:
            op_idx = s.rfind(operator, 0, idx)
            if op_idx > prev_operator:
                prev_operator = op_idx
        start_idx = prev_operator
        return start_idx+1, s[start_idx+1:idx]
    #if start_idx>0:
    #    #return start_idx, s[start_idx+1:idx]
    #    return start_idx, s[start_idx:idx]
    #else: return 0, s[:idx]

def get_succeeding_factor(s, idx, open_to_close):
    # return end_idx, substr
    if s[idx+2]=="(":
        end_idx = open_to_close[idx+2]
        return end_idx+1, s[idx+2:end_idx+1]
    else:
        # Need to find the closest succeeding +,-,*,/
        next_operator = len(s)
        for operator in ["+", "-", "*", "/"]:
            op_idx = s.find(operator, idx+2)
            if op_idx<0: continue
            if op_idx < next_operator:
                next_operator = op_idx
        end_idx = next_operator
        return end_idx, s[idx+2:end_idx]

def get_matching_parens(s):
    opens = []
    pairs = []
    close_to_open=dict()
    open_to_close=dict()
    for i, char in enumerate(s):
        if char=="(": opens.append(i)
        if char==")":
            open_idx = opens.pop()
            pairs.append((open_idx, i))
            close_to_open[i]=open_idx
            open_to_close[open_idx]=i
    return pairs, close_to_open, open_to_close


def process(fname):
    outfile = open("%s.processed"%fname, "w")
    with open(fname) as infile:
        for line in infile:
            if len(line.strip())==0 or line.strip()[0]=="#":
                outfile.write(line)
                continue
            s = line

            
            s = process_string(s)
            
            outfile.write(s)
    outfile.close()


def process_string(s):
    while True:
        parens, close_to_open, open_to_close = get_matching_parens(s)
        idx = s.find("**")
        if idx<0: return s

        start_idx, base = get_preceeding_factor(s, idx, close_to_open)
        end_idx, exponent = get_succeeding_factor(s, idx, open_to_close)

        if False:
            print("\n**********\n")
            print(s)
            print(idx)
            print(start_idx, base)
            print(end_idx, exponent)
            print("np.power(%s,%s)"%(base, exponent))

        s = "%snp.power(%s,%s)%s"%(s[:start_idx], base, exponent, s[end_idx:])
        #print(s)

process("musyc.txt")
process("musyc_jacobian.txt")


#s = "b*(A+b*X)**(c*d-x**alpha*3)"
#s = "U=(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2+r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2+r1**(gamma21+1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12)"
#process_string(s)
