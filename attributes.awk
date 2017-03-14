BEGIN{FS=";";max=0}
{
  for(i = 4; i <= NF; i = i +2){
    if($i > max){
       max = $i;
    }
  }
}
END{

  print max;
}
