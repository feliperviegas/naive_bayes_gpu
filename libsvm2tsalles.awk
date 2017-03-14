BEGIN{docid=1;}
{
	printf docid";1;CLASS="$1;
	for(i=2;i<=NF;i++){
	  split($i, token, ":");
	  printf ";"token[1]";"token[2];
	}
	printf "\n";
	docid+=1;
}