import datetime
from typing import List

import pandas as pd
import pandasql as ps


class DataProcessor:
    """DataProcessor class for processing Stepik's Reports data and generating features.

    Parameters
    ----------
    submissions_path : str
        Path to the submissions data file.
    course_structure_path : str
        Path to the course structure data file.
    first_course_day : datetime.date
        The start date of the course.
    admin_ids : List[int]
        List of admin user IDs to exclude from the analysis.

    Attributes
    ----------
    submissions_path : str
        Path to the submissions data file.
    course_structure_path : str
        Path to the course structure data file.
    first_course_day : datetime.date
        The start date of the course.
    admin_ids : List[int]
        List of admin user IDs to exclude from the analysis.
    df : pd.DataFrame
        Merged DataFrame containing submissions and course structure data.
    users_days_info : pd.DataFrame
        DataFrame containing user-specific daily statistics.

    Methods
    -------
    load_data()
        Load and preprocess the data from submissions and course structure files.
    create_users_days_info()
        Create user-specific daily statistics.
    create_features(users_days_info)
        Generate features based on user-specific daily statistics.
    fit()
        Fit the data processor by loading data and generating features.
    """

    def __init__(self,
                 submissions_path: str,
                 course_structure_path: str,
                 first_course_day: datetime.date,
                 admin_ids: List[int]) -> None:
        self.submissions_path = submissions_path
        self.course_structure_path = course_structure_path
        self.first_course_day = first_course_day
        self.admin_ids = admin_ids
        self.df = None
        self.users_days_info = None


    def load_data(self) -> None:
        """Load and preprocess the data from submissions and course structure files.

        This method reads submissions and course structure data from the provided CSV files,
        merges them based on the 'step_id' column, and preprocesses the data by calculating
        the 'score' and 'day' columns. It also filters the data based on the start date of
        the course and excludes admin users specified in the 'admin_ids' attribute.
        """
        submissions = pd.read_csv(self.submissions_path)
        course_structure = pd.read_csv(self.course_structure_path)
        self.df = pd.merge(submissions, course_structure, on='step_id')
        self.df['score'] = self.df['score'] * self.df['step_cost']
        self.df['day'] = pd.to_datetime(self.df['submission_time'], unit='s').dt.date

        self.df = self.df[self.df['day'] >= self.first_course_day]
        self.df = self.df[~self.df['user_id'].isin(self.admin_ids)]

        cols = ['submission_id', 'step_id', 'user_id', 'day', 'status', 'score', 'step_cost']
        self.df = self.df[cols].reset_index(drop=True)


    @staticmethod
    def _calc_solved_tasks(user_day: pd.DataFrame) -> int:
        """Calculate the number of solved tasks for a user on a specific day.

        This static method takes a DataFrame containing a user's data for a specific day
        as input and calculates the number of tasks that were solved correctly with a
        full score ('correct' status and score equal to step_cost).

        Parameters
        ----------
        user_day : pd.DataFrame
            DataFrame containing user's data for a specific day.

        Returns
        -------
        int
            Number of solved tasks on the given day.
        """
        correct_status = user_day['status'] == 'correct'
        full_score = user_day['score'] == user_day['step_cost']
        return user_day[correct_status & full_score]['step_id'].nunique()


    def _calculate_user_days_stats(self) -> pd.DataFrame:
        """Calculate user-specific daily statistics for active days.

        This method computes user-specific daily statistics, which include the number of
        submissions ('n_submits'), the number of unique tasks attempted ('n_tasks'), and
        the number of tasks solved correctly with a full score ('n_solved_tasks') for each
        user on active days. Active days are those when a user interacted with the course.

        Returns
        -------
        pd.DataFrame
            DataFrame containing user-specific daily statistics for active days.
        """
        grouped_df = self.df.groupby(['user_id', 'day'])
        users_days_stats = grouped_df.agg(
            n_submits=('submission_id', 'count'),
            n_tasks=('step_id', 'nunique')
        )
        users_days_stats['n_solved_tasks'] = grouped_df.apply(self._calc_solved_tasks)
        return users_days_stats.reset_index()


    def create_users_days_info(self) -> None:
        """Create a comprehensive user-specific days information table.

        This method generates a comprehensive table that includes all days when a user had
        access to the course. For each user, starting from the first day they accessed the
        course, it includes information for both active days (calculated using
        '_calculate_user_days_stats') and inactive days (where the user did not interact
        with the course, filled with zeros).

        The resulting 'users_days_info' DataFrame contains user-specific daily statistics,
        including 'user_id,' 'day,' 'n_submits,' 'n_tasks,' and 'n_solved_tasks' for all
        days associated with the user.
        """
        users = self.df[['user_id']].drop_duplicates()
        days = self.df[['day']].drop_duplicates()
        users_days = pd.merge(users['user_id'], days['day'], how='cross')

        users_first_days = self.df.groupby(['user_id'], as_index=False).agg(
            first_day=('day', 'min')
        )
        users_days_stats = self._calculate_user_days_stats()

        users_days = pd.merge(users_days, users_first_days, on='user_id', how='left')
        users_days = users_days[users_days['day'] >= users_days['first_day']]
        users_days.drop(columns=['first_day'], inplace=True)

        self.users_days_info = pd.merge(users_days, users_days_stats,
                                        on=['user_id', 'day'], how='left').fillna(0)
        self.users_days_info.sort_values(by=['user_id', 'day'], inplace=True)
        self.users_days_info.reset_index(drop=True, inplace=True)


    def create_features(self, users_days_info: pd.DataFrame) -> None:
        """Create a feature DataFrame based on user-specific daily statistics.

        This method generates a feature DataFrame using SQL-like queries on 'users_days_info.'
        The resulting 'features_df' includes various features calculated from user-specific
        daily statistics, such as 'solved_total,' 'days_offline' (number of days since the last
        submission), 'avg_submits_14d' (average submission attempts over the last 14 days,
        including the current day), 'success_rate_14d' (the proportion of successful attempts
        over the last 14 days, including the current day), and 'target_14d' (whether the user
        was online for the next 14 days, where 1 indicates no online activity, and 0 indicates
        online activity).

        Parameters
        ----------
        users_days_info : pd.DataFrame
            DataFrame containing user-specific daily statistics.
        """
        query = """
        SELECT
            user_id,
            day,
            n_submits,
            n_tasks,
            n_solved_tasks,
            SUM(n_solved_tasks) OVER (PARTITION BY user_id ORDER BY day) as solved_total,
            (SUM(n_submits) OVER
            (PARTITION BY user_id ORDER BY day ROWS BETWEEN 13 PRECEDING and CURRENT ROW)) as sum_submits_14d,
            (SUM(n_solved_tasks) OVER
            (PARTITION BY user_id ORDER BY day ROWS BETWEEN 13 PRECEDING and CURRENT ROW)) as sum_solved_14d,
            (SUM(n_tasks) OVER
            (PARTITION BY user_id ORDER BY day ROWS BETWEEN 13 PRECEDING and CURRENT ROW)) as sum_tasks_14d,
            (SUM(n_submits) OVER
            (PARTITION BY user_id ORDER BY day ROWS BETWEEN 1 FOLLOWING AND 14 FOLLOWING)) as sum_submits_future14d,
            (MAX(CASE WHEN n_submits > 0 THEN day ELSE NULL END) OVER
            (PARTITION BY user_id ORDER BY day)) as last_active_day
        FROM users_days_info
        """

        features_df = ps.sqldf(query)

        features_df['day'] = pd.to_datetime(features_df['day'], format='%Y-%m-%d')
        features_df['last_active_day'] = pd.to_datetime(features_df['last_active_day'], format='%Y-%m-%d')

        features_df['days_offline'] = (features_df['day'] - features_df['last_active_day']).dt.days
        features_df['avg_submits_14d'] = (features_df['sum_submits_14d'] / 14).round(2)
        features_df['success_rate_14d'] = (features_df['sum_solved_14d'] / features_df['sum_tasks_14d'])
        features_df['success_rate_14d'] = features_df['success_rate_14d'].fillna(0).round(2)
        features_df['target_14d'] = (features_df['sum_submits_future14d'] == 0).astype(int)

        features_list = ['user_id', 'day', 'solved_total', 'days_offline',
                         'success_rate_14d', 'avg_submits_14d', 'target_14d']
        self.features_df = features_df[features_list]


    def fit(self) -> None:
        """Fit the DataProcessor by loading data, creating user-specific days information, and generating features.

        This method prepares the DataProcessor for analysis by performing the following steps:
        1. Loads data using the 'load_data' method to populate the DataFrame.
        2. Creates user-specific daily statistics using the 'create_users_days_info' method.
        3. Generates features based on user-specific statistics using the 'create_features' method.

        This comprehensive process allows the DataProcessor to be ready for further analysis and
        modeling tasks.
        """
        self.load_data()
        self.create_users_days_info()
        self.create_features(self.users_days_info)
