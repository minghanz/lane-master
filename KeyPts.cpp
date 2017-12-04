#include "KeyPts.h"

void KeyPts::renew(vector<Point>& key_left_2p_new, vector<Point>& key_left_2n_new, vector<Point>& key_right_2p_new, vector<Point>& key_right_2n_new)
{
    key_left_2p = key_left_2p_new;
    key_left_2n = key_left_2n_new;
    key_right_2p = key_right_2p_new;
    key_right_2n = key_right_2n_new;

    renew_flag = true;
    
}
// void KeyPts::match()
// {
//     int left_match_num, right_match_num;
//     left_match_num = min(key_left_2p.size(), key_left_2p_last.size());
//     right_match_num = min(key_right_2p.size(), key_right_2p_last.size());
    
//     for (int i = 0; i < )
// }

// void match_side(vector<Point>& key_2p, vector<Point>& key_2p_last, int match_num )
// {
//     if (match_num >= 3)
//     {

//     }
// }