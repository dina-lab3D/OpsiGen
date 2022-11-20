#!/usr/bin/perl -w

use strict;

if ($#ARGV!=1) {
  print "Usage: align.pl <first_pdb> <second_pdb>\n";
  exit;
}

my $first_pdb = $ARGV[0];
my $second_pdb = $ARGV[1];

`/cs/staff/dina/utils/match.linux $first_pdb $second_pdb > match`
