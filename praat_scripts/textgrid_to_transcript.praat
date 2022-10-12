# Will Styler
# ERROR: line 20- found string expression instead of a numeric expression
# TextGrid information extraction script
# Gathers labels from other tiers on Textgrids
# So if you want to find the label from another tier corresponding to the time of another tier


grid$ = selected$ ("TextGrid")
resultfile$ = "'grid$'_info.txt"

header_row$ = "#"+ tab$ + "start"+ " " + "end" + " " + "addressee" + " " + "transcript" + " " + newline$
fileappend "'resultfile$'" 'header_row$'


selectObject: "TextGrid 'grid$'"

numint = Get number of intervals... 2
# Start the loop
for i from 1 to numint
	lab$ = ""
	selectObject: "TextGrid 'grid$'"
	lab$ = Get label of interval: 2, 'i'
	if lab$ <> ""
		vstart = Get start point: 2, 'i'
		vend = Get end point: 2, 'i'
		vdur = vend - vstart
		midpoint = vstart + (vdur/2)

		# Spit the results into a text file
		int1 = Get interval at time... 1 'midpoint'
		lab1$ = Get label of interval... 1 int1
		# print "'vdur'"
		prevint = int1 - 1
		result_row$ =  "'vstart'" + " " + "'vend'" + " " + "'lab$'" + ": " + "'lab1$'" + " " + newline$ + newline$
		fileappend "'resultfile$'" 'result_row$'
	endif	
endfor