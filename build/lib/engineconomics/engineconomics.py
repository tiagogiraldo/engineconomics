#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

from pandas import DataFrame
from numpy import exp, log
from plotly.express import bar
from scipy.optimize import root

class factor(object):
    '''
    Engineering economics factors
    '''
    
    def pgivenfsp(self, i:float, n:int)->float:
        '''
        Single-payment present worth factor. Find Present Worth (P) given  
        a Future worth (F).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.    
        '''
        self.i = i
        self.n = n
        return 1 /((1 + self.i)**(self.n))

    def fgivenpsp(self, i:float, n:int)->float:
        '''
        Single-payment compound amount factor. Find Future Worth (F) given 
        a Present Worth (P).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.            
        '''
        self.i = i
        self.n = n
        return (1 + self.i)**self.n

    def pgivena(self, i:float, n:int)->float:
        '''
        Uniform series present worth.  Find Present Worth (P) given
        a Uniform Series (A) of cash flows.

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.            
        '''
        self.i = i
        self.n = n
        return ((1 + self.i)**(self.n) - 1)/(self.i*(1+self.i)**self.n)
    
    def agivenp (self, i:float, n:int)->float:
        '''
        Capital recovery. Find a Uniform Series (A) given a Present
        Worth(P).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.            
        '''        
        self.i = i
        self.n = n
        return (self.i*(1+self.i)**self.n)/((1 + self.i)**(self.n) - 1)

    def fgivena (self, i, n):
        '''
        Uniform series compound amount.  Find a Future Worth (F) given
        a Uniform Serie (A).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.            
        '''
        self.i = i
        self.n = n
        return ((1+self.i)**self.n - 1)/self.i

    def agivenf (self, i:float, n:int)->float:
        '''
        Sinking fund.  Find a Uniform Serie (A) given a Future
        Worth (F).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.            
        '''        
        self.i = i
        self.n = n
        return self.i/((1+self.i)**self.n - 1)
    
    def pgivenag (self, i:float, n:int)->float:
        '''
        Arithmetic Gradient series present worth. Find a Present 
        Worth (P) given an Arithmetic Gradient Serie (G).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.            
        '''        
        self.i = i
        self.n = n
        return(((1+self.i)**self.n) - self.i*self.n - 1)/((self.i**2) * (((1+self.i)**self.n)))

    def fgivenag(self, i:float, n:int)->float:
        '''
        Arithmetic gradient series future worth. Find a future Worth (F) given
        an Arithmetic Gradient Serie (G).        

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.            
        '''        
        self.i = i
        self.n = n
        return (1/self.i)*((((1+self.i)**n-1)/self.i)-n)


    def agivenag(self, i:float, n:int)->float:
        '''
        Arithmetic gradient to equal payment series.  Find a Unifom Serie (A) given
        an Arithmetic Gradient Serie (G).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods            
        '''        
        self.i = i
        self.n = n
        return (1 / self.i) - (self.n /((1 + self.i)**self.n - 1))

    def pgivenga1(self, i:float, n:int, g:float)->float:
        '''
        Geometric gradient series present worth. Find a Present 
        Worth (P) given an Geometric Gradient Serie (G).

        Input arguments:
            i:   Interest rate per (uniform) period.
            n:   Number of uniform interest periods.
            g:   Geometric gradient or constant percentage or constant growth.           
        '''        
        self.i = i
        self.n = n
        self.g = g

        if (self.g != self.i):
            return (1 - (((1 + self.g)/(1 + self.i))**self.n))/(self.i - self.g)
        else:
            return self.n / (1+self.i) 



class time_value(object):
      

    def cfv(self, CF: float, F: str, i: float, n: float, g: float=None) -> float:
        '''
        "P/F": Find P Present Worth given F Future worth, interest i and number of periods n.
        "F/P": Find F Future worth given P Present Worth, interest i and number of periods n., 
        "P/A": Find P Present Worth given A Equal payment series, interest i and number of periods n.
        "A/P": Find A Equal payment series given P Present Worth, interest i and number of periods n.  
        "F/A": Find F Future worth given A Equal payment series, interest i and number of periods n. 
        "A/F": Find A Equal payment series given F Future worth, interest i and number of periods n.   
        "P/G": Find P Present Worth given G Arithmetic Gradient, interest i and number of periods n.
        "P/g"Find P Present Worth given g Geometric Gradient, A1 First payment, interest i and number of periods n.
        '''
        cf_asked = {
            "P/F": "PV", 
            "F/P": "FV", 
            "P/A": "PV", 
            "A/P": "A", 
            "F/A": "FV", 
            "A/F": "A", 
            "P/G": "PV",
            "F/G": "FV",
            "A/G": "A",
            "P/g": "PV"
            }

        cf_given = {
            "P/F": "FV", 
            "F/P": "PV", 
            "P/A": "A", 
            "A/P": "PV", 
            "F/A": "A", 
            "A/F": "FV", 
            "P/G": "G",
            "F/G": "G",
            "A/G": "G",            
            "P/g": "A1"
            }

        self.CF = CF
        self.F  = F
        self.i  = i
        self.n  = n
        self.g  = g


        self.values = {}
        self.values[cf_given[self.F]] = self.CF
        self.values['Factor'] = self.F
        self.values['i'] = self.i
        self.values['n'] = self.n
        if self.g is not None:
            self.values['g'] = self.g



        factor_list = ["P/F", "F/P", "P/A", "A/P", "F/A", "A/F", "P/G", "F/G", "A/G","P/g"]
        
        
        try:

            if self.F in factor_list:
  
                if self.F == "P/F":
                    value = self.CF * factor.pgivenfsp(self, self.i, self.n)
                elif self.F == "F/P":
                    value = self.CF * factor.fgivenpsp(self,self.i, self.n)
                elif self.F == "P/A":
                    value = self.CF * factor.pgivena(self, self.i, self.n)
                elif self.F == "A/P":
                    value = self.CF * factor.agivenp(self, self.i, self.n)
                elif self.F == "F/A":
                    value = self.CF * factor.fgivena(self, self.i, self.n)
                elif self.F == "A/F":
                    value = self.CF * factor.agivenf(self, self.i, self.n)
                elif self.F == "P/G":
                    value = self.CF * factor.pgivenag(self, self.i, self.n)
                elif self.F == 'F/G':
                    value = self.CF * factor.fgivenag(self, self.i, self.n)
                elif self.F == 'A/G':
                    value = self.CF * factor.agivenag(self, self.i, self.n)            
                elif self.F == "P/g":
                    if g is None:
                        print ('Input geometric gradient')
                    else:
                        value = self.CF * factor.pgivenga1(self, self.i, self.n, self.g) 
                
                self.values[cf_asked[self.F]] = value      
                return self.values            
        
        except:
            raise Exception("Check arguments")



    def pvp(self, a:float, i:float)->float:
        '''
        Perpetual present value.  Find Present Worth (P) given
        a Perpetual Uniform Serie (A).

        Input arguments:
            a: Perpetual Uniform Serie.
            i: Interest rate per (uniform) period.        
        '''

        self.a = a
        self.i = i
        return self.a/self.i

    
    def _getcf(self, period, cf_tuples_list):
        '''
        This function is used to filter the tuples (p, cf) of a list 
        that share the same first element (p) of the tuple and 
        calculate the sum of the cash flows (cf) corresponding to 
        the filtered element (p). 
        
        Input arguments:
            period:  Period to filter
            cf_tuples_list: List of tuples (p, cf)
        '''
    
        self.period = period
        self.cf_tuples_list = cf_tuples_list

        cf_tuples  = list(filter(lambda x:self.period in x, self.cf_tuples_list)) 
        
        if cf_tuples:
            pcf = 0
            for p, tcf in cf_tuples:
                pcf += tcf
            return (p, pcf)
        else:
            return (self.period, 0)   
    
    def npv(self, period_list:list, cf_list:list, i:float)->float:
        '''
        This function is used to estimate the net present value 
        from a list of cash flows, a list of the corresponding 
        periods to calculate the present value and an effective 
        interest rate.

        Input arguments:
            period_list: Period list
            cf_list: Cash flow list
            i: Effective interest rate
        '''
        self.period_list = period_list
        self.cf_list = cf_list
        self.i = i

        p_len = len(self.period_list)
        cf_len = len(self.cf_list)
        assert p_len == cf_len, f"The length of the period list ({p_len}) must be equal to the length of the cash flow list ({cf_len})." 
        #assert i > 0, f"Interest rate {i} must be greater than 0"  

        n_max=max(self.period_list) + 1 
        CFL = list(zip(self.period_list, self.cf_list))

        ncf = []

        for p in range (n_max):
            self.p = p
            self.cf_tuples_list=CFL
            cf_tuples = time_value._getcf(self, self.p, self.cf_tuples_list)
            ncf.append(cf_tuples)

        dcf = [x[1] / (1+i)**x[0] for x in ncf ]
        npv_ = sum(dcf)
        
        return npv_


    def npviv(self, cf_list:list, iv:list)->float:
        '''
        This function estimates the net present value for several cash 
        flows with different effective rates for each period.  In this 
        case there should be only one cash flow and one interest rate 
        for each period.  For the calculation to be consistent, the spacing 
        between periods must be uniform (monthly, bimonthly, annually, etc.) 
        and the effective interest rates for each period must correspond 
        to the same periodicity.

        Input arguments:
            cf_list
            iv
        '''
        
        len_cf_list = len(cf_list)
        len_iv = len(iv)
        
                
        assert len_cf_list == len_iv, f"The cash flow list {(len_cf_list)} and interest rates list{(len_iv)} has not the same number of elements"
        

        
        self.period_list = list(range(len_iv))
        self.cf_list = cf_list
        self.iv = iv
        
        n_max=max(self.period_list) + 1 

        CFL = list(zip(self.period_list, self.cf_list))

                
        i = []

        i_comp =1
        
        for r in self.iv:
            i_comp *= (1+r)

            i.append(i_comp)
            
        
        ncf = []
     
        for p in range(n_max):
            self.p = p
            self.cf_tuples_list=CFL
            _, cf = time_value._getcf(self, self.p,self.cf_tuples_list)
            ncf.append((p, cf, i[p]))

        dcf = [x[1] / x[2] for x in ncf ]
        npv_ = sum(dcf)
        
        self.period_list[0] = self.period_list[0] + npv_

        return npv_ 



    def vpn_terminal_value(self, cf_n: float, r:float, g:float, n:float)->float:
        '''
        This function returns two values.  The first is the terminal value in period n. 
        The second returns the present value of this terminal value in period 0.

        Input arguments:
            cf_n : Cash flow at the end of period n
            r: Discount cash flow rate at year n
            g: Growth rate
            n: Discount valuation period
        '''
        
        assert r != g, f"The interest rate {r} must be diferent from growth {g}"

        self.cf_n = cf_n
        self.r = r
        self.g = g
        self.n = n
        self.tval = (self.cf_n * ( 1 + self.g)) / (self.r - self.g)
        self.tvpv = self.tval * factor.pgivenfsp(self, self.r, self.n)

        return self.tval, self.tvpv


    def vpn_terminal_value_variable_rates(self, cf_n:float, rate_list:list, g:float)->float:
        '''
        This function returns two values.  The first is the terminal value in period n. 
        The second returns the present value of this terminal value in period 0.

        Input arguments:
            cf_n : Cash flow at the end of period n
            r: Discount cash flow rate at year n
            g: Growth rate
        '''
        
        len_r = len(rate_list)
        r = rate_list[len_r-1]

        assert r != g, f"The interest rate {r} must be diferent from growth {g}"
        
        self.r = r
        self.cf_n = cf_n
        self.r_list = rate_list
        self.g = g
        tval = (self.cf_n * ( 1 + self.g)) / (self.r - self.g)
        cf_list = [0] * len_r
        cf_list[len_r-1] = tval
        tvpv = time_value.npviv(self, cf_list, self.rate_list) 

        return tval, tvpv

class time_value_table(object):

    def cfdataframe(self, cf_dic:dict):
        '''
        
        '''
        
        self.cf_dic= cf_dic

        if self.cf_dic == None:
            return None

        if (self.cf_dic['Factor'] == 'P/F') or (self.cf_dic['Factor'] == 'F/P'):
            
            pv = -round(self.cf_dic['PV'], 4)
            fv =  round(self.cf_dic['FV'], 4)             

            n = self.cf_dic['n']
            x_data = range(n+1)
            y_o_data = [pv] + [0.] * (n)
            y_i_data = [0.] * (n) + [fv]

            return DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])

            
        if (self.cf_dic['Factor'] == 'P/A') or (self.cf_dic['Factor'] == 'A/P'):
            
            pv = -round(self.cf_dic['PV'], 4)
            a = round(self.cf_dic['A'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_i_data = [0.] + [a] * (n)
            y_o_data = [0.] * (n+1)
            y_o_data[0] = pv 
            return DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])  
                 

        if (self.cf_dic['Factor'] == 'F/A') or (self.cf_dic['Factor'] == 'A/F'):
            
            fv = round(self.cf_dic['FV'], 4)
            a = -round(self.cf_dic['A'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_o_data = [0.] + [a] * (n)
            y_i_data = [0.] * (n+1)
            y_i_data[-1] = fv
            return DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])

        if (self.cf_dic['Factor'] == 'P/G'):

            pv = -round(self.cf_dic['PV'], 4)
            ag = round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[0] = pv 
            return DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            
    
        if (self.cf_dic['Factor'] == 'A/G'):

            a = -round(self.cf_dic['A'], 4)
            ag = round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] + [a] * (n)
            return DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            


        if (self.cf_dic['Factor'] == 'F/G'):

            fv = round(self.cf_dic['FV'], 4)
            ag = -round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[-1] = fv 
            return DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Outcome", "Income"])            



        if (self.cf_dic['Factor'] == 'P/g'):

            pv = -round(self.cf_dic['PV'], 4)
            gg = round(self.cf_dic['g'], 4)
            ba = round(self.cf_dic['A1'], 4)
            n = cf_dic['n']
            x_data = range(n+1)

            y_i_data = [0] + [ba * (1 +  gg)**k   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[0] = pv 
            return DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            

    def npvtable(self, period_list:list, cf_list:list, i:float):

        self.period_list = period_list
        self.cf_list = cf_list
        self.i = i

        p_len = len(self.period_list)
        cf_len = len(self.cf_list)
        assert p_len == cf_len, f"The length of the period list ({pl}) must be equal to the length of the cash flow list ({cf_len})." 
        assert i > 0, f"Interest rate {i} must be greater than 0"  



        n_max=max(self.period_list) + 1 
        CFL = list(zip(self.period_list, self.cf_list))
        
        ncfl = []
        
        
        for p in range(n_max):
            cf_tuples  = list(filter(lambda x:p in x, CFL))    
            if cf_tuples:
                income = 0
                outcome = 0
                for n in range(len(cf_tuples)):
                    if cf_tuples[n][1] > 1:
                        income += cf_tuples[n][1]*1.0
                        outcome += 0.
                    else:
                        income += 0.
                        outcome += cf_tuples[n][1]*1.0
                ncf = income + outcome
                self.n = p
                pv = ncf * factor.pgivenfsp(self, self.i,self.n)
                ncfl.append((p, outcome, income, ncf, pv))
            else:
                ncfl.append((p, 0, 0, 0, 0))
                
        return DataFrame(ncfl, columns=['Period', 'Outcome', 'Income', 'ncf', 'dcf'])

    def npvsensitivitytable(self, period_list:list, cf_list:list, i_list:list):
        '''     
        This function allows you to calculate different net present 
        values for the same cash flow from a list of different interest 
        rates.

        Input arguments:
            period_list: Period list
            cf_list: Cash flow list
            i_list: Effective interest rate list
        '''

        self.period_list = period_list
        self.cf_list = cf_list
        self.i_list = i_list


        self.i_list.sort()
        
        table = [(time_value.npv(self.period_list, self.cf_list, r), r) for r in self.i_list]
        
        return  DataFrame(table, columns=['npv', 'i'])

    def npvivtable(self, period_list:list, cf_list:list, iv:list):
        '''
        Input arguments:
            period_list: 
            cf_list
            iv
        '''
        
        len_period_list = len(period_list)
        len_cf_list = len(cf_list)
        len_iv = len(iv)
        
        for p in period_list:
            c = period_list.count(p)
            if c > 1:
                raise Exception(f"There should only be one cash flow per period.  Period {p} has {c} elements")
                
        assert len_period_list == len_cf_list and len_period_list == len_iv, f"The inpt list has not the same number of elements"
        

        
        self.period_list = period_list
        self.cf_list = cf_list
        self.iv = iv
        
        n_max=max(period_list) + 1 
        CFL = list(zip(self.period_list, self.cf_list))

                
        i = []

        i_comp =1
        
        for r in self.iv:
            i_comp *= (1+r)

            i.append(i_comp)
            
        
        ncf = []
        income = 0
        outcome = 0
        for p in range (n_max):
            self.p = p
            self.cf_tuples_list=CFL
            _, cf = time_value._getcf(self, self.p,self.cf_tuples_list)
            
            if cf>0:
                income = cf
                outcome = 0
            else:
                income =  0
                outcome = cf

            if p ==0:
                ie = (i[p]**(1))-1
            else:
                ie = (i[p]**(1/p))-1
            dcf = cf/((1+ie)**p)
            ncf.append((p, income, outcome, cf, i[p], ie, dcf))

        df = DataFrame(ncf, columns=["Period", 'Outcome','Income',"ncf", "i_comp", "ie", "dcf"])
        return df


    def uniform_loan_amortization(self, loan_amount:float, rate:float, loan_term:int, periodicity:str='Y'):
        per_conv = ['M', 'B', 'Q','S', 'Y']
        per_names = {'M':'Month', 'B':'Bimonth', 'Q':'Quarter','S': 'Semiannual', 'Y':'Year'}
        assert periodicity in per_conv, f"Input 'Y' for Year, 'S' for Semiannual, 'Q' for Quarterly,  'B' for Bimonthly and 'M' for Monthly"
        self.loan_amount= loan_amount
        self.rate = rate
        self.loan_term = loan_term
        self.periodicity = periodicity

        per = list(range(self.loan_term + 1))
        beg_bal = []
        pay_per = []
        pri_per = []
        int_per = []
        tot_pay = []
        tot_int = []
        rem_bal = []

        period_payment = round(self.loan_amount * factor.agivenp(self, i=rate, n=loan_term), 2)

        pay_per.append(0)
        beg_bal.append(loan_amount)
        pri_per.append(0)
        int_per.append(0)
        tot_pay.append(0)
        tot_int.append(0)
        rem_bal.append(loan_amount)

        for p in range(1, self.loan_term + 1):
            beginning_balance = rem_bal[p-1]
            period_interest = round(rem_bal[p-1] * self.rate, 2)
            period_principal = period_payment - period_interest
            total_principal = sum(pri_per) + period_principal
            total_interest = sum(int_per) + period_interest
            remaining_balance  = beginning_balance - period_principal

            
            beg_bal.append(beginning_balance)
            pay_per.append(period_payment)
            pri_per.append(period_principal)
            int_per.append(period_interest)
            tot_pay.append(total_principal)
            tot_int.append(total_interest)
            rem_bal.append(remaining_balance)            

        
        data = zip(per, beg_bal, pay_per, pri_per, int_per, tot_pay, tot_int, rem_bal)


        period  = per_names[self.periodicity]
        columns = [period, 'Beginning balance', 'Payment', 'Principal', 'Interest', 'Total Payment', 'Total Interest', 'Remaining Balance' ]

        amortization_table = DataFrame(data=data, columns=columns)

        return amortization_table



class time_value_plot(object):
    
    def cf_plot_bar(self, cf_dic:dict):
        
        self.cf_dic = cf_dic

        if self.cf_dic == None:
            self.cf_dic = {}
        else:
            self.cf_dic = cf_dic



        if (self.cf_dic['Factor'] == 'P/F') or (self.cf_dic['Factor'] == 'F/P'):
            
            pv = -round(self.cf_dic['PV'], 4)
            fv =  round(self.cf_dic['FV'], 4)             

            n = self.cf_dic['n']
            x_data = range(n+1)
            y_o_data = [pv] + [0] * (n)
            y_i_data = [0.] * (n) + [fv]
            df = DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])
            print(df)

            if  (self.cf_dic['Factor'] == 'P/F'):
                title = f"{fv: .4f} * (P/F, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-pv:.4f}"
            else:
                title = f"{-pv: .4f} * (F/P, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {fv:.4f}"

            fig = bar(df, 
                      x="Period", 
                      y= ["Income", "Outcome"],
                      title= title , 
                      text_auto=True, 
                      opacity=0.80, 
                      facet_col_spacing= 0.0
                      )
            
            y_max = round(fv,0) + 0.5
            y_min = round(-pv, 0) + 0.5
            pp = y_min / (y_min + y_max)
            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_layout(yaxis_range=[-y_min, y_max])
            
            return(fig.show())



        if (self.cf_dic['Factor'] == 'P/A') or (self.cf_dic['Factor'] == 'A/P'):
            
            pv = -round(self.cf_dic['PV'], 4)
            a = round(self.cf_dic['A'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_i_data = [0.] + [a] * (n)
            y_o_data = [0.] * (n+1)
            y_o_data[0] = pv 
            df = DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])            
            print(df)

            if (self.cf_dic['Factor'] == 'P/A'):
                title = f"{a: .4f} * (P/A, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-pv:.4f}"
            else:
                title = f"{-pv: .4f} * (A/P, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {a:.2f}"
                
        
            fig = bar(
                df, 
                x="Period", 
                y=["Income", "Outcome"],
                title= title , 
                text_auto=True, 
                opacity=0.80, 
                facet_col_spacing= 0.0,
            )
            
            y_max = round(a,0) + 0.5
            y_min = round(-pv, 0) + 0.5
            pp = y_min / (y_min + y_max)
                

            fig.update_layout(bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_yaxes(title="Cash Flow [$]")                            
            fig.update_layout(yaxis_range=[-y_min, y_max])
            fig.update_yaxes(title="Cash Flow [$]")     

            return(fig.show())

        if (self.cf_dic['Factor'] == 'F/A') or (self.cf_dic['Factor'] == 'A/F'):
            
            fv = round(self.cf_dic['FV'], 4)
            a = -round(self.cf_dic['A'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_o_data = [0.] + [a] * (n)
            y_i_data = [0.] * (n+1)
            y_i_data[-1] = fv
            df = DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])
            print(df)
            
            if (self.cf_dic['Factor'] == 'F/A'):
                title = f"{-a: .4f} * (F/A, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {fv:.4f}"
            else:
                title = f"{fv: .4f} * (A/F, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-a:.4f}"
            
            
            fig = bar(df, 
                    x="Period", 
                    y=["Income", "Outcome"],
                    title= title ,                  
                    text_auto=True,
                    opacity=0.80, 
                    facet_col_spacing= 0.0,
                    )
            
            y_max = round(fv,0) + 10
            y_min = round(-a, 0) + 10
            pp =  y_min / ( y_min + y_max)
                

            fig.update_layout(bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_yaxes(title="Cash Flow [$]")                            
            fig.update_layout(yaxis_range=[-y_min, y_max])
            return(fig.show())


        if (self.cf_dic['Factor'] == 'P/G'):

            pv = -round(self.cf_dic['PV'], 4)
            ag = round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[0] = pv 
            df = DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            
            print(df)
            
            
            title = f"{ag:.2f} * (P/G, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-pv:.4f}"            
            fig = bar(
                df, 
                x="Period", 
                y=["Gradient Income", "Outcome"],
                title= title , 
                text_auto=True, 
                opacity=0.80, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(ag * (n-1) * 1.5,0) 
            y_min = round(-pv, 0) + 0.5
            pp = y_min / (y_min + y_max)
 
            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_yaxes(title="Cash Flow [$]")       
            fig.update_layout(yaxis_range=[-y_min, y_max])
            
            return(fig.show())


        if (self.cf_dic['Factor'] == 'A/G'):

            a = -round(self.cf_dic['A'], 4)
            ag = round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] + [a] * (n)
            df = DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            
            print(df)
            
            
            title = f"{ag:.2f} * (A/G, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-a:.4f}"            
            fig = bar(
                df, 
                x="Period", 
                y=["Outcome", "Gradient Income"],
                title= title , 
                text_auto=True, 
                opacity=0.80, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(ag * (n-1) * 1.5,0) 
            y_min = round(-a, 0) + 0.5
            pp = y_min / (y_min + y_max)
 

            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_yaxes(title="Cash Flow [$]")       
            fig.update_layout(yaxis_range=[-y_min, y_max])
            
            return(fig.show())

        if (self.cf_dic['Factor'] == 'F/G'):

            fv = round(self.cf_dic['FV'], 4)
            ag = -round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[-1] = fv 
            df = DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Outcome", "Income"])            
            print(df)
            
            
            title = f"{-ag:.2f} * (F/G, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {fv:.4f}"            
            fig = bar(
                df, 
                x="Period", 
                y=["Income", "Gradient Outcome"],
                title= title , 
                text_auto=True, 
                opacity=0.80, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(fv, 0) + 0.5
            y_min = round(-ag * (n-1) * 1.5,0) 
            pp = y_min / (y_min + y_max)
 
            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_yaxes(title="Cash Flow [$]")       
            fig.update_layout(yaxis_range=[-y_min, y_max])
            
            return(fig.show())

        if (self.cf_dic['Factor'] == 'P/g'):

            pv = -round(self.cf_dic['PV'], 4)
            gg = round(self.cf_dic['g'], 4)
            ba = round(self.cf_dic['A1'], 4)
            n = cf_dic['n']
            x_data = range(n+1)

            y_i_data = [0] + [round(ba * (1 +  gg)**k,4)  for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[0] = round(pv,4)
            df = DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            
            print(df)
            
            
            title = f"{ba: .2f} * (P/g, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}, g: {gg*100:.2f}) = {-pv:.4f}"

            fig = bar(
                df, 
                x="Period", 
                y=["Gradient Income", "Outcome"],
                title= title , 
                text_auto=True, 
                opacity=0.80, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(ba * (1+gg)**(n-1) * 1.2,0 )
            y_min = round(-pv, 0) + 0.5
            pp = y_min / (y_min + y_max)
    

            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_layout(yaxis_range=[-y_min, y_max])
            fig.update_yaxes(title="Cash Flow [$]")       
            
            return(fig.show()) 
    
    def __arrowupdate(df_x, df_y):    

        counter = 0
        arrow_list = []
        text_list = []

        for i in df_y.tolist():
            if i != 0:
                if i>0:
                    arrowcolor = 'rgb(77,7,252)'
                    xanchor = 'right'
                    text = round(i,2)
                    textangle = 0
                    visible=True                     
                else:
                    arrowcolor = 'rgb(252,7,77)'
                    xanchor = 'left'
                    text = round(i,2)
                    textangle = 0
                    visible=True                     

                arrow_aux=dict(x=df_x.values[counter],
                            y=df_y.values[counter],
                            xref='x',
                            ax=counter,
                            ay=0,
                            yref='y',
                            axref='x',
                            ayref='y',
                            text='',
                            showarrow=True,
                            arrowhead=1,
                            arrowsize=2,
                            arrowwidth=2,
                            arrowcolor=arrowcolor,
                            xanchor = xanchor,
                            yanchor='bottom'
                            )            

                text_aux = dict(text = text,
                                textangle = textangle, 
                                visible=visible, 
                                )

                arrow_list.append(arrow_aux)
                text_list.append(text_aux)

                counter += 1
            else:
                counter += 1    

        return arrow_list, text_list

    def cf_plot_arrow(self, cf_dic):
    
        
        if self.cf_dic == None:
            self.cf_dic = {}
        else:
            self.cf_dic = cf_dic



        if (self.cf_dic['Factor'] == 'P/F') or (self.cf_dic['Factor'] == 'F/P'):
          
            pv = -round(self.cf_dic['PV'], 4)
            fv =  round(self.cf_dic['FV'], 4)             

            n = self.cf_dic['n']
            x_data = range(n+1)
            y_o_data = [pv] + [0.] * (n+1)
            y_i_data = [0.] * (n) + [fv]

            df= DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])
            print(df)

            if  (self.cf_dic['Factor'] == 'P/F'):
                title = f"{fv: .4f} * (P/F, i: {self.cf_dic['i']*100:.2f}%  n: {self.cf_dic['n']}) = {-pv:.4f}"
            else:
                title = f"{-pv: .4f} * (F/P, i: {self.cf_dic['i']*100:.2f}%  n: {self.cf_dic['n']}) = {fv:.4f}"


            fig = bar(df, 
                        x="Period", 
                        y=["Income", "Outcome"],
                        title= title , 
                        text_auto=False, 
                        opacity=0.0, 
                        facet_col_spacing= 0.0,
                        )
            
            y_max = round(fv,0) + 5
            y_min = round(-pv, 0) + 5
            pp = y_min / (y_min + y_max)

            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )

            fig.update_layout(yaxis_range=[-y_min, y_max])

            arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(df['Period'], df["Income"])
            arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(df['Period'], df["Outcome"])

            fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
            fig.update_layout(annotations=text_list_1 + text_list_2)  
            fig.update_traces(textposition='inside')            

            return(fig.show()) 


        if (self.cf_dic['Factor'] == 'P/A') or (self.cf_dic['Factor'] == 'A/P'):
            
            pv = -round(self.cf_dic['PV'], 4)
            a = round(self.cf_dic['A'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_i_data = [0.] + [a] * (n)
            y_o_data = [0.] * (n+1)
            y_o_data[0] = pv 
            df = DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])            
            print(df)

            if (self.cf_dic['Factor'] == 'P/A'):
                title = f"{a: .4f} * (P/A, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-pv:.4f}"
            else:
                title = f"{-pv: .4f} * (A/P, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {a:.2f}"
                
        
            fig = bar(
                df, 
                x="Period", 
                y=["Income", "Outcome"],
                title= title , 
                text_auto=False, 
                opacity=0.0, 
                facet_col_spacing= 0.0,
            )
            
            y_max = round(a,0) + 0.5
            y_min = round(-pv, 0) + 0.5
            pp = y_min / (y_min + y_max)               

            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )

            fig.update_layout(yaxis_range=[-y_min, y_max])

            arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(df['Period'], df["Income"])
            arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(df['Period'], df["Outcome"])

            fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
            fig.update_layout(annotations=text_list_1 + text_list_2)  
            fig.update_traces(textposition='inside')    

            return(fig.show())             

        if (self.cf_dic['Factor'] == 'F/A') or (self.cf_dic['Factor'] == 'A/F'):
            
            fv = round(self.cf_dic['FV'], 4)
            a = -round(self.cf_dic['A'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_o_data = [0.] + [a] * (n)
            y_i_data = [0.] * (n+1)
            y_i_data[-1] = fv
            df = DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Income", "Outcome"])
            print(df)
            
            if (self.cf_dic['Factor'] == 'F/A'):
                title = f"{-a: .4f} * (F/A, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {fv:.4f}"
            else:
                title = f"{fv: .4f} * (A/F, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-a:.4f}"
            
            
            fig = bar(df, 
                    x="Period", 
                    y=["Income", "Outcome"],
                    title= title , 
                    text_auto=False, 
                    opacity=0.0, 
                    facet_col_spacing= 0.0
                    )
            
            y_max = round(fv,0) + 0.5
            y_min = -round(a, 0) + 0.5
            pp = y_min / (y_min + y_max)
                

            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_layout(yaxis_range=[-y_min, y_max])

            arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(df['Period'], df["Income"])
            arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(df['Period'], df["Outcome"])

            fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
            fig.update_layout(annotations=text_list_1 + text_list_2)  
            fig.update_traces(textposition='inside')    
                
            return(fig.show())


        if (self.cf_dic['Factor'] == 'P/G'):

            pv = -round(self.cf_dic['PV'], 4)
            ag = round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[0] = pv 
            df = DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            
            print(df)
            
            
            title = f"{ag:.2f} * (P/G, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-pv:.4f}"            

            fig = bar(
                df, 
                x="Period", 
                y=["Gradient Income", "Outcome"],
                title= title , 
                text_auto=False, 
                opacity=0.0, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(ag * (n-1) * 1.5,0) 
            y_min = round(-pv, 0) + 0.5
            pp = y_min / (y_min + y_max)

      
            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_layout(yaxis_range=[-y_min, y_max])

            arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(df['Period'], df["Gradient Income"])
            arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(df['Period'], df["Outcome"])

            fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
            fig.update_layout(annotations=text_list_1 + text_list_2)  
            fig.update_traces(textposition='inside')    

            return(fig.show())


        if (self.cf_dic['Factor'] == 'A/G'):

            a = -round(self.cf_dic['A'], 4)
            ag = round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] + [a] * (n)
            df = DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            
            print(df)
            
            
            title = f"{ag:.2f} * (A/G, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {-a:.4f}"            
            fig = bar(
                df, 
                x="Period", 
                y=["Outcome", "Gradient Income"],
                title= title , 
                text_auto=False, 
                opacity=0.0, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(ag * (n-1) * 1.5,0) 
            y_min = round(-a, 0) + 0.5
            pp = y_min / (y_min + y_max)
 

            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_yaxes(title="Cash Flow [$]")       
            fig.update_layout(yaxis_range=[-y_min, y_max])
            
            arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(df['Period'], df["Gradient Income"])
            arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(df['Period'], df["Outcome"])

            fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
            fig.update_layout(annotations=text_list_1 + text_list_2)  
            fig.update_traces(textposition='inside')   


            return(fig.show())


        if (self.cf_dic['Factor'] == 'F/G'):

            fv = round(self.cf_dic['FV'], 4)
            ag = -round(self.cf_dic['G'], 4)
            n = cf_dic['n']
            x_data = range(n+1)
            y_ag_data = [0.] + [k * ag   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[-1] = fv 
            df = DataFrame(list(zip(x_data, y_ag_data, y_o_data)), columns=["Period", "Gradient Outcome", "Income"])            
            print(df)
            
            
            title = f"{-ag:.2f} * (F/G, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}) = {fv:.4f}"            
            fig = bar(
                df, 
                x="Period", 
                y=["Income", "Gradient Outcome"],
                title= title , 
                text_auto=False, 
                opacity=0.0, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(fv, 0) + 0.5
            y_min = round(-ag * (n-1) * 1.5,0) 
            pp = y_min / (y_min + y_max)
 
            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_yaxes(title="Cash Flow [$]")       
            fig.update_layout(yaxis_range=[-y_min, y_max])

            arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(df['Period'], df["Gradient Outcome"])
            arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(df['Period'], df["Income"])

            fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
            fig.update_layout(annotations=text_list_1 + text_list_2)  
            fig.update_traces(textposition='inside')               
            
            return(fig.show())

        if (self.cf_dic['Factor'] == 'P/g'):

            pv = -round(self.cf_dic['PV'], 4)
            gg = round(self.cf_dic['g'], 4)
            ba = round(self.cf_dic['A1'], 4)
            n = cf_dic['n']
            x_data = range(n+1)

            y_i_data = [0] + [round(ba * (1 +  gg)**k,4)   for k in range(n)]
            y_o_data = [0.] * (n+1)
            y_o_data[0] = round(pv,4)
            df = DataFrame(list(zip(x_data, y_i_data, y_o_data)), columns=["Period", "Gradient Income", "Outcome"])            
            print(df)
            
            
            title = f"{ba: .2f} * (P/g, i: {self.cf_dic['i']*100:.2f}%,  n: {self.cf_dic['n']}, g: {gg*100:.2f}) = {-pv:.4f}"

            fig = bar(
                df, 
                x="Period", 
                y=["Gradient Income", "Outcome"],
                title= title , 
                text_auto=False, 
                opacity=0.0, 
                facet_col_spacing= 0.0
            )
            
            y_max = round(ba * (1+gg)**(n-1) * 1.2,0 )
            y_min = round(-pv, 0) + 0.5
            pp = y_min / (y_min + y_max)
    

            fig.update_layout( bargap=0.05,bargroupgap=0.75)
            fig.update_traces(textangle=90, selector=dict(type='bar'))
            fig.update_xaxes(position= pp, 
                            anchor="free", 
                            linecolor = "black", 
                            tickfont=dict(size=12, color='black'),
                            ticks = "outside",
                            ticklabelposition =  "outside left"
                            )
            fig.update_layout(yaxis_range=[-y_min, y_max])
            fig.update_yaxes(title="Cash Flow [$]")       

            arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(df['Period'], df["Gradient Income"])
            arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(df['Period'], df["Outcome"])

            fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
            fig.update_layout(annotations=text_list_1 + text_list_2)  
            fig.update_traces(textposition='inside')               
            
            return(fig.show()) 

    def npvplotbar(self, npvtable, i):
        '''
        Plot nominal cash flow stream from pandas data frame with npv
        '''
        # npvtable = npvtable[['Period', 'Income', 'Outcome']]
        self.npvtable = npvtable
        npv_ = self.npvtable['dcf'].sum()

        if i==None:
            cf_list = list(self.npvtable['ncf'])
            period_list = list(self.npvtable['Period'])
            i = compound_interest.irr(self, npw=npv_,period_list=period_list,cf_list=cf_list)
            title = f"NPV: {npv_: .2f}, irr: {i: .4f}"
        else:
            title = f"NPV: {npv_: .2f}, i: {i: .4f}"


        fig = bar(
            npvtable, 
            x="Period", 
            y=["Income", "Outcome"],
            title= title , 
            text_auto=False, 
            opacity=0.8, 
            facet_col_spacing= 0.0
        )

        y_max = round(self.npvtable['ncf'].max() + 10)
            
        if self.npvtable['ncf'].min() < 0: 
            y_min = round(self.npvtable['ncf'].min() + 10) * -1.
        else:
            y_min =  y_max

        pp = y_min / (y_min + y_max)


        fig.update_layout( bargap=0.05,bargroupgap=0.75)
        fig.update_traces(textangle=90, selector=dict(type='bar'))
        fig.update_xaxes(position= pp, 
                         anchor="free", 
                         linecolor = "black", 
                         tickfont=dict(size=12, color='black'),
                         ticks = "outside",
                         ticklabelposition =  "outside left"
                        )
        fig.update_layout(yaxis_range=[-y_min, y_max])
        fig.update_yaxes(title="Cash Flow [$]")       
        
        return(fig.show())    

    def npvplotarrow(self, npvtable, i):
        '''
        Plot nominal cash flow stream from pandas data frame with npv
        '''
        # npvtable = npvtable[['Period', 'Income', 'Outcome']]        
        self.npvtable = npvtable

        npv_ = self.npvtable['dcf'].sum()

        if i==None:
            cf_list = list(self.npvtable['ncf'])
            period_list = list(self.npvtable['Period'])
            i = compound_interest.irr(self, npw=npv_,period_list=period_list,cf_list=cf_list)
            title = f"NPV: {npv_: .2f}, irr: {i: .4f}"
        else:
            title = f"NPV: {npv_: .2f}, i: {i: .4f}"

        fig = bar(
            self.npvtable, 
            x="Period", 
            y=["Income", "Outcome"],
            title= title , 
            text_auto=False, 
            opacity=0.0, 
            facet_col_spacing= 0.0
        )
        y_max = round(npvtable['ncf'].max() + 10)
            
        if self.npvtable['ncf'].min() < 0: 
            y_min = round(self.npvtable['Outcome'].min() + 10) * -1.
        else:
            y_min =  y_max

        pp = y_min / (y_min + y_max)


        fig.update_layout( bargap=0.05,bargroupgap=0.75)
        fig.update_traces(textangle=90, selector=dict(type='bar'))
        fig.update_xaxes(position= pp, 
                         anchor="free", 
                         linecolor = "black", 
                         tickfont=dict(size=12, color='black'),
                         ticks = "outside",
                         ticklabelposition =  "outside left"
                        )
        fig.update_layout(yaxis_range=[-y_min, y_max])
        fig.update_yaxes(title="Nominal Cash Flow [$]")       

        arrow_list_1, text_list_1 = time_value_plot.__arrowupdate(self.npvtable['Period'], self.npvtable["Income"])
        arrow_list_2, text_list_2 = time_value_plot.__arrowupdate(self.npvtable['Period'], self.npvtable["Outcome"])

        fig.update_layout(annotations=arrow_list_1 + arrow_list_2)  
        fig.update_layout(annotations=text_list_1 + text_list_2)  
        fig.update_traces(textposition='inside')               
        
        return(fig.show())        


class compound_interest(object):
    
    def spi(self, pv: float, fv: float, n: float)->float:
        '''
        spi: Single Payment Interest

        Input arguments:        
            pv: Present Value
            fv: Future Value
        '''
        
        self.pv = pv
        self.fv = fv
        self.n = n
        
        return ((self.fv/self.pv)**(1/self.n))-1

    def ei(self, r: float, m: float)->float:
        '''
        ei: Effective Interest Per Time Period
        
        Input arguments:
            r:  Nominal Interest Rate For Same Time Period
            m:  Number of Times Interest Is Compounded Per Stated Time Period
        '''
        self.r = r
        self.m = m
        return (1 + self.r / self.m)**self.m - 1
    
    def ipa(self, ipm:float)->float:
        '''
        ipa: Interest Paid In Advance Per Time Period

        Input arguments:
            ipm: Interest Paid at Maturity Per Time Period
        '''
        self.ipm = ipm
        return self.ipm/(1 + self.ipm)
    

    def ipm(self, ipa:float)->float:
        '''
        ipm: Interest Paid at Maturity Per Time Period

        Input arguments:
            ipa: Interest Paid In Advance Per Time Period
        '''
        self.ipa = ipa
        return self.ipa/(1+self.ipa)

    def ni(self, i: float, m: float)->float:
        '''
        ni: Nominal annual interest rate

        Input arguments:
            i:  Effective Interest Per Time Period
            m:  Number of Times Interest Is Compounded Per Stated Time Period
        '''
        self.i = i
        self.m = m
        return self.m * ((1+self.i)**(1/self.m) - 1)

    def cci(self, r)->float:
        '''
        cci: Effective compounded continuously interest rate from nominal interest rate

        Input arguments:
            r:  Nominal Interest Rate For Same Time Period
        '''
        self.r = r
        return exp(self.r) -1 

    def nci(self,i)->float:
        '''
        nci: Nominal interest rate from continous interest rate

        Input argument:
            i:  Effective Interest Per Time Period
        '''
        self.i = i
        return log(1 + self.i)

    def rir(self, nir, g)->float:
        '''
        rir: real interest rate

        Inputs arguments:
            nir: Nominal interest rate
            g: Inflation rate 
        '''
        self.nir = nir
        self.g = g
        return (1+self.nir)/(1+self.g)

    def nir(self, rir, g)->float:
        '''
        nir: nominal interest rate

        Input arguments:
            rr: Real interest rate
            g:  Inflation rate
        '''
        self.rir =rir 
        self.g = g
        return (1+self.rir)*(1+self.g)

    def irr(self,npw, period_list:list, cf_list:list):
        '''
        Estimates the internal rate of return for a given series of cash flows.
        
        Input arguments:
            period_list: Period list
            cf_list: Cash flow list
            i: Effective interest rate
        '''
        self.period_list = period_list
        self.cf_list = cf_list
        self.npw = npw
        

        f = lambda x: npw - time_value.npv(self,period_list, cf_list, x)

        irr_ = root(f, [0], tol=0.00000001)['x'][0]
        
        return irr_            