#import the necessary libraries
from os import error
from types import resolve_bases
import matplotlib.pyplot as plt
import re
import sys
import linecache

#checks for correct input format
if (len(sys.argv)!=2): 
    sys.exit("The format is 'python3 bottleneck_analysis.py <filename>'")
filename=sys.argv[1]

"""
The Basic idea here is to identfy the three most expensive kernels in your program
and then give you a rough idea where the basic bottle-neck is. This tool is based on
a part of the documentation for GP-GPUsim (MICRO 2012). Link to the above mentioned: http://gpgpu-sim.org/manual/index.php/Main_Page#:~:text=gpu_tot_sim_insn%20/%20wall_time-,Simple%20Bottleneck%20Analysis,-These%20performance%20counters

The following is the implimentation theme of the tool:
1. Make a dictionary with each thread launch's kernel_uid as key and line number as value.
2. Take out and all the necessary counters for bottle-neck analysis.
3. Subtract from the preceeding occurence as these are incremental, not kernel specific.
4. Group all the uids with same name together.
5. Get TOP-3 expensive kernels.
6. Plot the counters on the graph using plt for each kernel.

"""


#this function gets the uid and line for each kernel
def uid_line(file):
    uid_pattern=re.compile("kernel_launch_uid = (\d+)")
    res={}
    for i,line in enumerate(open(file)):
        for match in re.finditer(uid_pattern,line):
            capture_id=list(match.group(1))
            capture_id=int(''.join(capture_id))
            if capture_id not in res:
                res[capture_id]={}
            res[capture_id]["line"]=i
            res[capture_id]["uid"]=capture_id
    
    return res

#this function gets a particular figure from a particular kernel, given its starting line
def fetch_figure(fp,stat,kernel_line):
    line_no=kernel_line
    pattern=re.compile("^"+stat+"\s*=\s*(.*)")
    end_ker_pattern=re.compile("^.*END-of-Interconnect-DETAILS.*$")

    matcher=re.match(pattern,linecache.getline(fp,line_no))
    while(not bool(matcher)):
        if(bool(re.match(end_ker_pattern,linecache.getline(fp,line_no)))):
            raise Exception("There is no such metric, please check the white spaces and perform a spell check")
        line_no+=1
        matcher=re.match(pattern,linecache.getline(fp,line_no))

    figure=list(matcher.group(1))
    figure=''.join(figure)
    return figure

# this function add a particular metric that you spicify, to the record of each kernel launch
def add_metric_by_uid(fp,res,metric):
    for i in res:
        try:
            (res[i])[metric]=float(fetch_figure(fp,metric,(res[i])["line"]))
        except:
            try:
                (res[i])[metric]=fetch_figure(fp,metric,(res[i])["line"]).strip()
            except Exception as e :
                print(e)
                break

# this is a function that a list of metrics to record of each kernel launch
def add_metrics_list(fp,res,metrics):
    for i in metrics:
        add_metric_by_uid(fp,res,i)


#this function is for the metrics that are essentially counters i.e. they are not kernel specific.
#These counters are the ones that are increamented hence need to be subtracted with their previous value.
def norm_metric(res,metric):
    
    first_key=next(iter(res))
    if(isinstance(res[first_key][metric],str)):
        raise Exception("metrics which are inserted as strings cannot be adjusted")
    
    lis=[]
    for i in res:
        lis.append((res[i])[metric])
    for i in range(len(lis)-1,0,-1):
        lis[i]=lis[i]-lis[i-1]
    m=0
    for i in res:
        (res[i])[metric]=lis[m]
        m=m+1


#Above function but with a list
def norm_metric_list(res,metrics):
    for i in metrics:
        norm_metric(res,i)

#Used for grouping the launch records on the basis of the metric specified.  
def grpby_metric(res, metric):
    grp_res={}
    list_uni=[]
    for i in res:
        list_uni.append( (res[i])[metric] )
    list_uni=list(set(list_uni))
    for m in list_uni:
        for k in res:
            if(res[k][metric]==m):
                if m not in grp_res:
                    grp_res[m]=[]
                grp_res[m].append(res[k])
    return grp_res

#this kernel adds the cycles taken up by each of the kernel in totality and plots a graph. I also returns a list of the three most heavy kernels
def cycle_analysis():
    res=uid_line(filename)
    metrics = ["gpu_sim_cycle","kernel_name"]
    add_metrics_list(filename,res,metrics)
    grpd_res=grpby_metric(res,"kernel_name")
    view={}
    #add cycles of each instance to figure out the heaviest
    for i in grpd_res:
        prop=0
        for q in grpd_res[i]:
            prop=prop+q["gpu_sim_cycle"]
        view[i]=prop
    sorted_view={k:v for k,v in sorted(view.items(),key=lambda item: item[1])}
    
    x=[]
    y=[]
    for i in sorted_view:
        x.append(i)
        y.append(view[i])
    top3=[]
    #get the top3 heaviest
    for k in range(-1,-4,-1):
        top3.append(x[k])
    #plot the kernel graph
    plt.barh(x,y)
    plt.show()
    return top3

def plot_metric(metric,res,kn):
    x=[]
    y=[]
    for i in res:
        x.append(i[metric])
    #print(len(x))
    for i in range(0,len(x)):
        #print(i)
        y.append(i)
    plt.figure(figsize=(20,10))
    plt.bar(y,x)
    plt.title(kn+"::"+metric)
    plt.show() 


def bottle_neck_analysis(kers):
    res=uid_line(filename)
    metrics=["gpgpu_n_stall_shd_mem","gpu_stall_dramfull","gpu_stall_icnt2sh   ","kernel_name"]
    tbn    =["gpgpu_n_stall_shd_mem","gpu_stall_dramfull","gpu_stall_icnt2sh   "]
    add_metrics_list(filename,res,metrics)
    norm_metric_list (res,tbn)
    #print(res)
    gprd_res=grpby_metric(res,"kernel_name")
    tot=0
    pl_x=[]
    pl_y=[]
    for i in gprd_res:
        pl_x.append(i)
        pl_y.append(len(gprd_res[i]))
        tot=tot+len(gprd_res[i])
    plt.barh(pl_x,pl_y)

   
    for i in kers:
        for m in tbn:
            plot_metric(m,gprd_res[i],i)
    return gprd_res



top3=cycle_analysis()
bottle_neck_analysis(top3)
