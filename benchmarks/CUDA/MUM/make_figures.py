#!/fs/sz-user-supported/Linux-i686/bin/python2.5
import matplotlib
matplotlib.use('PS')

import pylab
import csv

def get_stats(filename):
	stats = {}
	statfile = open(filename)
	stats = dict([(key, float(value)) for (key, value) in csv.reader(statfile)])
	return stats

from pylab import arange,pi,sin,cos,sqrt

def set_figure_props():
	fig_width_pt = 225.0  # Get this from LaTeX using \showthe\columnwidth
	inches_per_pt = 1.0/72.27               # Convert pt to inch
	golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
	fig_width = fig_width_pt*inches_per_pt  # width in inches
	fig_height = 1.5*fig_width*golden_mean      # height in inches
	fig_size =  [fig_width,fig_height]
	params = {'backend': 'ps',
			  'axes.labelsize': 8,
			  'axes.linewidth': 0.5,
			  'text.fontsize': 8,
			  'xtick.labelsize': 7,
			  'ytick.labelsize': 7,
			  'legend.fontsize': 7,
			  'legend.linewidth':0.5,
			  'title.fontsize' : 8,
			  'text.usetex': True,
			  'figure.figsize': fig_size}
	pylab.rcParams.update(params)

def draw_speedup_figures(outfile, fig_title):
	query_lens = []
	app_speedups = []
	kernel_speedups = []
	f = open(outfile)
	headers = f.next()
	headers = headers.strip()
	headers = headers.split(',')
	
	query_col = headers.index("QUERY")
	kernel_col = headers.index("KERNEL_SPEEDUP")
	mummer_speed_col = headers.index("MUMMER_SPEEDUP")
	
	for vals in csv.reader(f, 'excel', delimiter=' '):
		query_lens.append(int(vals[query_col]))
		app_speedups.append(float(vals[mummer_speed_col]))
		kernel_speedups.append(float(vals[kernel_col]))
		
	draw_speedup_fig(query_lens,
					 kernel_speedups,
					 fig_title,
					 outfile + ".kernel_speedup.eps")

def draw_speedup_fig(x, y, fig_title, filename):
	set_figure_props()
	ax = pylab.subplot(111)
	
	pylab.semilogx(x, y, linestyle=':', marker='v', basex=2)

	pylab.xticks(x)
	frm = pylab.FormatStrFormatter("%d")
	ax.xaxis.set_major_formatter(frm)

	ax.xaxis.grid(True, which="minor")
	pylab.xlabel("Query length (bp - log scale)")
	pylab.ylabel("Speedup")
	pylab.title(fig_title, fontsize=9)

	pylab.savefig(filename)
	pylab.close()
	
def make_time_breakout():
	statfiles = ["cbriggsae/cleanreads.fna-100.gpustats",
				 "lmonocytogenes/cleanreads.fna-20.gpustats",
				 "s_suis/cleanreads.fna-20.gpustats"
				 ]
	
	labels = [ "\emph{C. briggsae}",
			   "\emph{L. monocytogenes}",
			  "\emph{S. suis} "
			  ]
	
	stats = {}
	convert_to_seconds = ["Total",
						  "Kernel",
						  "Print matches",
						  "Copy queries to GPU",
						  "Copy output from GPU",
						  "Copy suffix tree to GPU",
						  "Read queries from disk",
						  "Suffix tree constructions"]
	
	for f in statfiles:
		f_stats = get_stats(f)
		for (key, value) in f_stats.iteritems():
			
			if key in convert_to_seconds:
				val = value / 1000.0 #float( value/f_stats["TOTAL"]
			else:
				val = int(value)
			if key in stats:
				stats[key].append(val)
			else:
				stats[key] = [val]			
	
	ind = arange(0,3.6,1.2 )    # the x locations for the groups

	width = 0.35       # the width of the bars: can also be len(x) sequence
	
	i = 0	
## colors = [ "#e31a1c",
## 		   "#377db8",
## 		   "#4daf4a",
## 		   "#984ea3",
## 		   "#ffff33",
## 		   "#ff7f00"]
	
	colors = [ "#FF4500",
			   "#1E90FF",
			   "#90EE90",
			   "#FFD700",
			   "#DA70D6",
			   "#D2B48C"]

	set_figure_props()
	pylab.subplot(111)
	transfer = []
	for j in range(0, len(statfiles)):
		transfer.append( stats["Copy suffix tree to GPU"][j] + stats["Copy output from GPU"][j] + stats["Copy queries to GPU"][j])

	stats["Data transfer to GPU"] = transfer
	
	del stats["Copy suffix tree to GPU"]
	del stats["Copy output from GPU"]
	del stats["Copy queries to GPU"]

	del stats["Total"]
	
	lengths = stats["Minimum substring length"]
	
	del stats["Minimum substring length"]
	del stats["Average query length"]


	plots = []
	running_totals = [0 for j in range(0,len(statfiles))]
	for (category, series) in stats.iteritems():
		plots.append(pylab.bar(ind, series, width, color=colors[i], bottom=running_totals))
		running_totals = [running_totals[j] + series[j] for j in range(0, len(series))]
		i += 1
	
	#pylab.xticks(ind+width/2., labels )
	pylab.xticks(ind +width/2., labels )
	

	pylab.title("Time spent by phase in MUMmerGPU", fontsize=9)
	pylab.ylabel("time (s)")
	pylab.xlim(-width,len(ind))
	pylab.ylim(0, 600)
	pylab.legend( [p[0] for p in reversed(plots)], [key for key in reversed(stats.keys())] )
	
	pylab.savefig('time_breakout.eps')
	pylab.close()

make_time_breakout()
draw_speedup_figures("anthrax/speedup.out", "Kernel speedup, GPU vs. CPU")
